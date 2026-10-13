"""Local derivative checks for the opt-in selected-face branch extension."""

from typing import cast

import pytest
import torch

from advar.transport import (
    BoundaryEdges,
    _volume_parts,
    finite_volume_step,
    finite_volume_trajectory,
    selected_face_extension,
    selected_face_extensions,
)


def test_selected_face_keeps_native_values_and_derivatives_on_its_own_side() -> None:
    dtype = torch.float64
    q = torch.tensor([[-0.4, 0.0, 0.7]], dtype=dtype, requires_grad=True)
    for column, side in ((0, -1), (2, 1)):
        with selected_face_extension("x", 0, column, side):
            plus, minus = _volume_parts(q, axis="x")
        native = _volume_parts(q, axis="x")
        torch.testing.assert_close(plus, native[0], atol=0, rtol=0)
        torch.testing.assert_close(minus, native[1], atol=0, rtol=0)

        # On the selected face's native side the extension is the same map,
        # including its first and second derivatives.
        score = (plus.square() + 2 * minus.square()).sum()
        grad = torch.autograd.grad(score, q, create_graph=True, retain_graph=True)[0]
        hvp = torch.autograd.grad(grad.sum(), q, retain_graph=True)[0]
        native_score = (native[0].square() + 2 * native[1].square()).sum()
        native_grad = torch.autograd.grad(
            native_score, q, create_graph=True, retain_graph=True,
        )[0]
        native_hvp = torch.autograd.grad(native_grad.sum(), q, retain_graph=True)[0]
        torch.testing.assert_close(grad, native_grad, atol=0, rtol=0)
        torch.testing.assert_close(hvp, native_hvp, atol=0, rtol=0)


@pytest.mark.parametrize("side, expected", [(-1, -3.0), (1, -1.0)])
def test_tie_uses_requested_donor_branch_when_neighbor_values_differ(side, expected):
    dtype = torch.float64
    echo = torch.tensor([[1.0, 3.0]], dtype=dtype)
    support = torch.ones_like(echo)
    qx = torch.zeros((1, 3), dtype=dtype, requires_grad=True)
    qy = torch.zeros((2, 2), dtype=dtype)

    def updated_cell(volume_flux: torch.Tensor) -> torch.Tensor:
        local_qx = qx.clone()
        local_qx[0, 1] = volume_flux
        result = finite_volume_step(
            echo, support, local_qx, qy,
            dt_seconds=0.1,
            spacing_yx=(1.0, 1.0),
            reconstruction="donorcell",
        )
        return result.echo[0, 0]

    with selected_face_extension("x", 0, 1, side):
        derivative = torch.autograd.grad(updated_cell(qx[0, 1]), qx)[0][0, 1]
    assert derivative.item() == pytest.approx(0.1 * expected, abs=1e-14)


def test_selected_face_is_axis_local_nested_and_restored_after_exception():
    values = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    other_axis = torch.tensor([[0.0, 2.0], [-2.0, 0.0]], dtype=torch.float64)
    with selected_face_extension("x", 0, 1, -1):
        x_parts = _volume_parts(values, axis="x")
        y_parts = _volume_parts(other_axis, axis="y")
        selected_grad = torch.autograd.grad(
            x_parts[0][0, 1], values, retain_graph=True,
        )[0]
        other_face_grad = torch.autograd.grad(
            x_parts[0][1, 1], values, retain_graph=True,
        )[0]
        torch.testing.assert_close(selected_grad[0, 1], torch.tensor(0.0, dtype=values.dtype))
        torch.testing.assert_close(other_face_grad[1, 1], torch.tensor(0.5, dtype=values.dtype))
        for actual, expected in zip(y_parts, _volume_parts(other_axis, axis="y")):
            torch.testing.assert_close(actual, expected)
        with selected_face_extension("x", 0, 1, 1):
            inner = _volume_parts(values, axis="x")
            inner_grad = torch.autograd.grad(
                inner[0][0, 1], values, retain_graph=True,
            )[0]
            torch.testing.assert_close(
                inner_grad[0, 1], torch.tensor(1.0, dtype=values.dtype),
            )
        restored = _volume_parts(values, axis="x")
        restored_grad = torch.autograd.grad(
            restored[0][0, 1], values, retain_graph=True,
        )[0]
        torch.testing.assert_close(
            restored_grad[0, 1], torch.tensor(0.0, dtype=values.dtype),
        )

    native = _volume_parts(values, axis="x")
    with pytest.raises(RuntimeError, match="leave"):
        with selected_face_extension("x", 0, 1, -1):
            _volume_parts(values, axis="x")
            raise RuntimeError("leave context")
    after_exception = _volume_parts(values, axis="x")
    after_exception_grad = torch.autograd.grad(
        after_exception[0][0, 1], values,
    )[0]
    torch.testing.assert_close(
        after_exception_grad[0, 1], torch.tensor(0.5, dtype=values.dtype),
    )
    torch.testing.assert_close(after_exception[0], native[0])


@pytest.mark.parametrize(
    "args, error, match",
    [
        (("z", 0, 0, 1), ValueError, "axis"),
        (("x", -1, 0, 1), ValueError, "row and column"),
        (("x", 0.5, 0, 1), TypeError, "row and column"),
        (("x", True, 0, 1), TypeError, "row and column"),
        (("x", 0, 0, 0), ValueError, "side"),
        (("x", 0, 0, True), ValueError, "side"),
    ],
)
def test_selected_face_rejects_invalid_context_arguments(args, error, match):
    with pytest.raises(error, match=match):
        with selected_face_extension(*args):
            pass


@pytest.mark.parametrize("axis, row, column", [("x", 1, 0), ("y", 0, 2)])
def test_selected_face_rejects_indices_outside_matching_flux(axis, row, column):
    with selected_face_extension(axis, row, column, 1):
        with pytest.raises(ValueError, match="outside the face-flux shape"):
            _volume_parts(torch.zeros((1, 2), dtype=torch.float64), axis=axis)


@pytest.mark.parametrize("side1", [-1, 1])
@pytest.mark.parametrize("side2", [-1, 1])
def test_two_selected_faces_choose_independent_zero_flux_derivatives(side1, side2):
    dtype = torch.float64
    q = torch.zeros((2, 3), dtype=dtype, requires_grad=True)
    selections = (("y", 0, 1, side1), ("y", 1, 2, side2))
    with selected_face_extensions(*selections):
        plus, minus = _volume_parts(q, axis="y")
        score = (2 * plus + 3 * minus).sum()
        gradient = torch.autograd.grad(score, q)[0]

    torch.testing.assert_close(plus, torch.zeros_like(q), atol=0, rtol=0)
    torch.testing.assert_close(minus, torch.zeros_like(q), atol=0, rtol=0)
    expected1 = 3.0 if side1 < 0 else 2.0
    expected2 = 3.0 if side2 < 0 else 2.0
    assert gradient[0, 1].item() == expected1
    assert gradient[1, 2].item() == expected2
    assert gradient[0, 0].item() == 2.5


def test_two_face_context_is_order_independent_unique_and_restored():
    q = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    first = (("y", 0, 1, -1), ("y", 1, 2, 1))
    outputs = []
    for faces in (first, tuple(reversed(first))):
        with selected_face_extensions(*faces):
            plus, minus = _volume_parts(q, axis="y")
            outputs.append((plus.clone(), minus.clone()))
    for actual, expected in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    with pytest.raises(ValueError, match="exactly two"):
        with selected_face_extensions(("y", 0, 1, -1)):
            pass
    with pytest.raises(ValueError, match="distinct faces"):
        with selected_face_extensions(("y", 0, 1, -1), ("y", 0, 1, 1)):
            pass
    with pytest.raises(ValueError, match="side"):
        with selected_face_extensions(("y", 0, 1, -1), ("y", 1, 2, 0)):
            pass

    restored = _volume_parts(q, axis="y")
    restored_gradient = torch.autograd.grad(restored[0][0, 1], q)[0]
    assert restored_gradient[0, 1].item() == 0.5


def _two_face_trajectory(*, coefficients: torch.Tensor, replay: bool):
    dtype = coefficients.dtype
    y, x = torch.meshgrid(
        torch.arange(3, dtype=dtype),
        torch.arange(3, dtype=dtype),
        indexing="ij",
    )
    echo = 1.0 + 0.05 * x.square() + 0.03 * y.square() + 0.02 * x * y
    support = torch.ones_like(echo)
    basis = torch.zeros((2, 4, 4), dtype=dtype)
    basis[0, 1, 1] = 1.0
    basis[1, 1, 3] = 1.0
    zero_echo = cast(BoundaryEdges, tuple(torch.zeros(3, dtype=dtype) for _ in range(4)))
    known_support = cast(BoundaryEdges, tuple(torch.ones_like(edge) for edge in zero_echo))
    return finite_volume_trajectory(
        echo,
        support,
        coefficients,
        torch.tensor(0.0, dtype=dtype),
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=0.05,
        spacing_yx=(1.0, 1.0),
        boundary_echo=((zero_echo, zero_echo),),
        boundary_support=((known_support, known_support),),
        reconstruction="donorcell",
        replay=replay,
    )[0]


def test_two_selected_faces_are_captured_for_replay_gradient_and_hvp():
    dtype = torch.float64
    objective_weights = torch.tensor(
        [[0.3, 0.8, 0.7], [1.2, 0.5, 0.4], [0.9, 0.6, 1.1]],
        dtype=dtype,
    )
    for side1 in (-1, 1):
        for side2 in (-1, 1):
            results = []
            for replay in (False, True):
                coefficients = torch.zeros(2, dtype=dtype, requires_grad=True)
                faces = (("y", 1, 1, side1), ("y", 1, 2, side2))
                with selected_face_extensions(*faces):
                    frames = _two_face_trajectory(coefficients=coefficients, replay=replay)
                objective = (frames[-1] * objective_weights).sum()
                gradient = torch.autograd.grad(objective, coefficients, create_graph=True)[0]
                hvp = torch.autograd.grad(gradient.sum(), coefficients)[0]
                results.append((frames.detach(), gradient.detach(), hvp.detach()))
            for actual, expected in zip(results[1], results[0]):
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            assert bool(torch.isfinite(results[1][1]).all())
            assert bool(torch.isfinite(results[1][2]).all())
            assert results[1][1].abs().sum().item() > 0
            assert results[1][2].abs().sum().item() > 0


def _trajectory(
    *, coefficient: torch.Tensor, replay: bool, reconstruction: str = "donorcell",
):
    dtype = coefficient.dtype
    y, x = torch.meshgrid(
        torch.arange(3, dtype=dtype),
        torch.arange(3, dtype=dtype),
        indexing="ij",
    )
    echo = 1.0 + 0.05 * x.square() + 0.03 * y.square() + 0.02 * x * y
    support = torch.ones_like(echo)
    basis = torch.zeros((1, 4, 4), dtype=dtype)
    basis[0, 1, 2] = 1.0
    zero_echo = cast(BoundaryEdges, tuple(torch.zeros(3, dtype=dtype) for _ in range(4)))
    known_support = cast(BoundaryEdges, tuple(torch.ones_like(edge) for edge in zero_echo))
    schedule_echo = ((zero_echo, zero_echo),)
    schedule_support = ((known_support, known_support),)
    return finite_volume_trajectory(
        echo,
        support,
        coefficient,
        torch.tensor(0.0, dtype=dtype),
        psi_basis=basis,
        leads=1,
        substeps_per_interval=1,
        interval_seconds=0.05,
        spacing_yx=(1.0, 1.0),
        boundary_echo=schedule_echo,
        boundary_support=schedule_support,
        reconstruction=reconstruction,
        replay=replay,
    )[0]


def test_trajectory_replay_freezes_extension_for_gradient_and_hvp_after_scope_closes():
    dtype = torch.float64
    results = []
    for replay in (False, True):
        coefficient = torch.tensor([0.0], dtype=dtype, requires_grad=True)
        with selected_face_extension("y", 1, 1, -1):
            frames = _trajectory(coefficient=coefficient, replay=replay)
        objective = (frames[-1] * torch.tensor(
            [[0.3, 0.8, 0.7], [1.2, 0.5, 0.4], [0.9, 0.6, 1.1]],
            dtype=dtype,
        )).sum()
        gradient = torch.autograd.grad(objective, coefficient, create_graph=True)[0]
        hvp = torch.autograd.grad(gradient, coefficient)[0]
        results.append((frames.detach(), gradient.detach(), hvp.detach()))

    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(results[1][1]) and torch.isfinite(results[1][2])
    assert results[1][1].abs().item() > 0
    assert results[1][2].abs().item() > 0


def test_native_replay_captures_none_when_backward_runs_inside_extension():
    dtype = torch.float64
    results = []
    for replay in (False, True):
        coefficient = torch.tensor([0.0], dtype=dtype, requires_grad=True)
        frames = _trajectory(
            coefficient=coefficient,
            replay=replay,
            reconstruction="minmod",
        )
        with selected_face_extension("y", 1, 1, -1):
            objective = (frames[-1] * torch.tensor(
                [[0.3, 0.8, 0.7], [1.2, 0.5, 0.4], [0.9, 0.6, 1.1]],
                dtype=dtype,
            )).sum()
            gradient = torch.autograd.grad(
                objective, coefficient, create_graph=True,
            )[0]
            hvp = torch.autograd.grad(gradient, coefficient)[0]
        results.append((frames.detach(), gradient.detach(), hvp.detach()))

    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert results[1][1].abs().item() > 0
    assert results[1][2].abs().item() > 0


def test_y_minmod_selected_face_matches_native_on_its_own_side():
    dtype = torch.float64
    y, x = torch.meshgrid(
        torch.arange(3, dtype=dtype),
        torch.arange(3, dtype=dtype),
        indexing="ij",
    )
    echo = 1.0 + 0.05 * x.square() + 0.03 * y.square() + 0.02 * x * y
    support = torch.ones_like(echo)
    psi = torch.zeros((4, 4), dtype=dtype)
    psi[1, 2] = 0.1
    qx = psi[1:, :] - psi[:-1, :]
    qy = -(psi[:, 1:] - psi[:, :-1])
    zero_echo = cast(BoundaryEdges, tuple(torch.zeros(3, dtype=dtype) for _ in range(4)))
    known_support = tuple(torch.ones(3, dtype=dtype) for _ in range(4))
    boundary_echo = (zero_echo, zero_echo)
    boundary_support = (known_support, known_support)

    def advance():
        return finite_volume_step(
            echo,
            support,
            qx,
            qy,
            dt_seconds=0.05,
            spacing_yx=(1.0, 1.0),
            boundary_echo=boundary_echo,
            boundary_support=boundary_support,
            max_courant=0.5,
            reconstruction="minmod",
        )

    native = advance()
    with selected_face_extension("y", 1, 1, -1):
        selected = advance()
    for name in (
        "echo",
        "support",
        "transport_inflow",
        "transport_outflow",
        "physical_inflow",
        "physical_outflow",
        "source_quadrature",
    ):
        torch.testing.assert_close(
            getattr(selected, name), getattr(native, name), atol=0, rtol=0,
        )
