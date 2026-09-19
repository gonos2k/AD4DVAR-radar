"""Focused contract tests for matrix-free replay."""

import pytest
import torch

from advar.matrix_free import recompute


def _loss(function, value: torch.Tensor) -> torch.Tensor:
    output = function(value)
    outputs = output if isinstance(output, tuple) else (output,)
    weights = torch.tensor([0.7, 1.1, 1.3], dtype=torch.float64)
    return sum((item * weights).pow(3).sum() for item in outputs)


@pytest.mark.parametrize(
    "function",
    [
        lambda value: value,
        lambda value: value.clone(),
        lambda value: value + 0,
        lambda value: (value, value),
    ],
    ids=["identity", "clone", "add_zero", "tuple_alias"],
)
def test_recompute_alias_forms_match_plain_grad_gradgrad_and_jvp_of_grad(
    function,
) -> None:
    value = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    direction = torch.tensor([-0.3, 0.5, -0.7], dtype=torch.float64)

    plain_grad = torch.func.grad(lambda item: _loss(function, item))
    replay_grad = torch.func.grad(
        lambda item: _loss(lambda inner: recompute(function, inner), item)
    )

    torch.testing.assert_close(replay_grad(value), plain_grad(value))

    plain_gradgrad = torch.func.grad(lambda item: plain_grad(item).sum())
    replay_gradgrad = torch.func.grad(lambda item: replay_grad(item).sum())
    torch.testing.assert_close(replay_gradgrad(value), plain_gradgrad(value))

    _, plain_jvp = torch.func.jvp(plain_grad, (value,), (direction,))
    _, replay_jvp = torch.func.jvp(replay_grad, (value,), (direction,))
    torch.testing.assert_close(replay_jvp, plain_jvp)


@pytest.mark.parametrize(
    "value",
    [torch.tensor([True, False]), torch.tensor([1, 2], dtype=torch.int64)],
    ids=["bool", "int"],
)
def test_recompute_rejects_nonfloating_inputs_before_function_execution(value):
    called = False

    def function(item):
        nonlocal called
        called = True
        return item

    with pytest.raises(TypeError, match="floating"):
        recompute(function, value)
    assert not called


def test_recompute_rejects_cpu_autocast_for_floating_input():
    value = torch.tensor([0.2, 0.4], dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="(?i)autocast"):
            recompute(lambda item: item + 1, value)
