"""Local, replay-off derivative extensions for one minmod stage event."""

import pytest
import torch
from typing import cast

from advar import transport


DTYPE = torch.float64


def _stage_inputs(axis: str, *, requires_grad: bool = False):
    if axis == "x":
        q = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            dtype=DTYPE,
        )
        qx = torch.full((3, 4), 0.1, dtype=DTYPE)
        qy = torch.zeros((4, 3), dtype=DTYPE)
    else:
        q = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            dtype=DTYPE,
        )
        qx = torch.zeros((3, 4), dtype=DTYPE)
        qy = torch.full((4, 3), 0.1, dtype=DTYPE)
    q.requires_grad_(requires_grad)
    edges = tuple(torch.zeros(3, dtype=DTYPE) for _ in range(4))
    return q, qx, qy, edges, q.new_tensor(0.1), q.new_tensor(1.0)


def _euler(q, qx, qy, edges, dt, area):
    return transport._euler_minmod(q, qx, qy, edges, dt, area)


def _trajectory(*, replay: bool):
    q = torch.tensor(
        [[1.0, 1.1, 1.2], [1.2, 1.3, 1.4], [1.4, 1.5, 1.6]], dtype=DTYPE,
    )
    support = torch.ones_like(q)
    basis = torch.zeros((1, 4, 4), dtype=DTYPE)
    basis[0, 1, 1] = 1.0
    zero_edges = cast(transport.BoundaryEdges,tuple(torch.zeros(3, dtype=DTYPE) for _ in range(4)))
    known_edges = cast(transport.BoundaryEdges,tuple(torch.ones(3, dtype=DTYPE) for _ in range(4)))
    return transport.finite_volume_trajectory(
        q, support, torch.tensor([0.01], dtype=DTYPE), torch.tensor(0.0, dtype=DTYPE),
        psi_basis=basis, leads=1, substeps_per_interval=2,
        interval_seconds=1.0, spacing_yx=(1.0, 1.0),
        boundary_echo=((zero_edges, zero_edges), (zero_edges, zero_edges)),
        boundary_support=((known_edges, known_edges), (known_edges, known_edges)),
        reconstruction="minmod", replay=replay,
    )


def test_positive_tie_extensions_keep_native_value_and_select_opposite_gradients():
    q, qx, qy, edges, dt, area = _stage_inputs("x", requires_grad=True)
    original = q.detach().clone()
    value_and_gradient = []
    for selector in (1, 2):
        with transport.selected_minmod_extension(0, "x", 1, 1, selector):
            result = _euler(q, qx, qy, edges, dt, area)
        value = result[1, 1]
        gradient, = torch.autograd.grad(value, q, retain_graph=True)
        value_and_gradient.append((value, gradient))

    native = _euler(q, qx, qy, edges, dt, area)
    assert torch.equal(value_and_gradient[0][0], value_and_gradient[1][0])
    assert torch.equal(value_and_gradient[0][0], native[1, 1])
    assert value_and_gradient[0][1][1, 2] == 0.0
    torch.testing.assert_close(
        value_and_gradient[1][1][1, 2], -0.5 * dt * qx[1, 2], rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(q.detach(), original, rtol=0.0, atol=0.0)


def test_global_stage_count_selects_requested_y_stage_and_restores_default():
    inputs = _stage_inputs("y")
    q, qx, qy, edges, dt, area = inputs
    native_first = _euler(q, qx, qy, edges, dt, area)
    native = q
    for _ in range(4):
        native = _euler(native, qx, qy, edges, dt, area)

    selected = q
    with transport.selected_minmod_extension(2, "y", 1, 1, 0):
        for _ in range(4):
            selected = _euler(selected, qx, qy, edges, dt, area)

    expected = q
    for stage in range(4):
        if stage == 2:
            with transport.selected_minmod_extension(0, "y", 1, 1, 0):
                expected = _euler(expected, qx, qy, edges, dt, area)
        else:
            expected = _euler(expected, qx, qy, edges, dt, area)
    torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(selected, native)
    after_scope = _euler(q, qx, qy, edges, dt, area)
    torch.testing.assert_close(after_scope, native_first, rtol=0.0, atol=0.0)


def test_selected_stage_runs_through_replay_off_trajectory():
    native, _ = _trajectory(replay=False)
    with transport.selected_minmod_extension(2, "x", 1, 1, 0):
        selected, _ = _trajectory(replay=False)
    assert torch.isfinite(selected).all()
    assert not torch.equal(selected, native)


def test_nested_extensions_share_stage_ordinal_and_restore_outer_selection():
    q = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=DTYPE,
    )
    qx = torch.full((3, 4), 0.1, dtype=DTYPE)
    qy = torch.full((4, 3), 0.1, dtype=DTYPE)
    edges = tuple(torch.zeros(3, dtype=DTYPE) for _ in range(4))
    dt, area = q.new_tensor(0.1), q.new_tensor(1.0)
    actual = q
    with transport.selected_minmod_extension(3, "x", 1, 1, 0):
        actual = _euler(actual, qx, qy, edges, dt, area)  # stage 0
        with transport.selected_minmod_extension(1, "y", 1, 1, 0):
            actual = _euler(actual, qx, qy, edges, dt, area)  # stage 1
        actual = _euler(actual, qx, qy, edges, dt, area)  # stage 2
        actual = _euler(actual, qx, qy, edges, dt, area)  # outer stage 3

    expected = q
    expected = _euler(expected, qx, qy, edges, dt, area)
    with transport.selected_minmod_extension(0, "y", 1, 1, 0):
        expected = _euler(expected, qx, qy, edges, dt, area)
    expected = _euler(expected, qx, qy, edges, dt, area)
    with transport.selected_minmod_extension(0, "x", 1, 1, 0):
        expected = _euler(expected, qx, qy, edges, dt, area)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_nested_same_event_replaces_outer_selector_and_restores_after_scope():
    q, qx, qy, edges, dt, area = _stage_inputs("x", requires_grad=True)
    native = _euler(q, qx, qy, edges, dt, area)
    with transport.selected_minmod_extension(0, "x", 1, 1, 1):
        with transport.selected_minmod_extension(0, "x", 1, 1, 2):
            selected = _euler(q, qx, qy, edges, dt, area)
    selected_gradient, = torch.autograd.grad(selected[1, 1], q, retain_graph=True)
    with transport.selected_minmod_extension(0, "x", 1, 1, 2):
        right = _euler(q, qx, qy, edges, dt, area)
    right_gradient, = torch.autograd.grad(right[1, 1], q, retain_graph=True)
    torch.testing.assert_close(selected_gradient, right_gradient, rtol=0.0, atol=0.0)
    restored = _euler(q, qx, qy, edges, dt, area)
    torch.testing.assert_close(restored, native, rtol=0.0, atol=0.0)


def test_inactive_zero_extension_has_finite_reverse_derivative():
    q, qx, qy, edges, dt, area = _stage_inputs("x", requires_grad=True)
    with transport.selected_minmod_extension(0, "x", 1, 1, 0):
        result = _euler(q, qx, qy, edges, dt, area)
    gradient, = torch.autograd.grad(result.square().sum(), q)
    assert torch.isfinite(result).all()
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("left_value, selectors", [(1.0, (2, 0)), (-1.0, (0, 2))])
def test_right_operand_zero_selects_the_correct_side(left_value, selectors):
    q = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0 - left_value, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=DTYPE,
        requires_grad=True,
    )
    qx = torch.full((3, 4), 0.1, dtype=DTYPE)
    qy = torch.zeros((4, 3), dtype=DTYPE)
    edges = tuple(torch.zeros(3, dtype=DTYPE) for _ in range(4))
    dt, area = q.new_tensor(0.1), q.new_tensor(1.0)
    values, gradients = [], []
    for selector in selectors:
        with transport.selected_minmod_extension(0, "x", 1, 1, selector):
            result = _euler(q, qx, qy, edges, dt, area)
        values.append(result[1, 1])
        gradient, = torch.autograd.grad(result[1, 1], q, retain_graph=True)
        gradients.append(gradient)

    assert torch.equal(values[0], values[1])
    assert gradients[0][1, 2] != gradients[1][1, 2]
    right_index = selectors.index(2)
    zero_index = selectors.index(0)
    torch.testing.assert_close(
        gradients[right_index][1, 2], -0.5 * dt * qx[1, 2], rtol=0.0, atol=0.0,
    )
    assert gradients[zero_index][1, 2] == 0.0
    assert all(torch.isfinite(value) for value in values)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_selected_extension_requires_replay_false_and_resets_after_error():
    with pytest.raises(ValueError, match="stage 0 was not reached"):
        with transport.selected_minmod_extension(0, "x", 1, 1, 1):
            with pytest.raises(ValueError, match="requires replay=False"):
                _trajectory(replay=True)

    # The guard is lexical and does not leak after the failed trajectory call.
    _trajectory(replay=True)


def test_unreached_stage_and_noninterior_stage_cell_are_rejected():
    q, qx, qy, edges, dt, area = _stage_inputs("x")
    with pytest.raises(ValueError, match="was not reached"):
        with transport.selected_minmod_extension(1, "x", 1, 1, 1):
            _euler(q, qx, qy, edges, dt, area)
    with pytest.raises(ValueError, match="must be interior"):
        with transport.selected_minmod_extension(0, "x", 0, 1, 1):
            _euler(q, qx, qy, edges, dt, area)


@pytest.mark.parametrize(
    "args, error, match",
    [
        ((True, "x", 1, 1, 1), TypeError, "stage"),
        ((-1, "x", 1, 1, 1), ValueError, "stage"),
        ((0, "z", 1, 1, 1), ValueError, "axis"),
        ((0, "x", True, 1, 1), TypeError, "row and column"),
        ((0, "x", 1, 1.5, 1), TypeError, "row and column"),
        ((0, "x", -1, 1, 1), ValueError, "nonnegative"),
        ((0, "x", 1, 1, True), TypeError, "selector"),
        ((0, "x", 1, 1, 3), ValueError, "selector"),
    ],
)
def test_selected_minmod_extension_rejects_invalid_arguments(args, error, match):
    with pytest.raises(error, match=match):
        with transport.selected_minmod_extension(*args):
            pass
