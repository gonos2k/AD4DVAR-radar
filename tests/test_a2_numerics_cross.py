import pytest
import torch

import advar.variational as variational
from advar.physics import RemapCell, remap, remap_core


def test_cross_dtype_giant_finite_remap_has_zero_connected_ad_paths() -> None:
    echo = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    displacement = torch.tensor((1.0e308, -1.0e308), dtype=torch.float64)

    def function(field: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        return remap(field, motion)

    output, tangent = torch.func.jvp(
        function,
        (echo, displacement),
        (torch.ones_like(echo), torch.ones_like(displacement)),
    )
    _, pullback = torch.func.vjp(function, echo, displacement)
    echo_gradient, displacement_gradient = pullback(torch.ones_like(output))
    echo_for_grad = echo.clone().requires_grad_()
    displacement_for_grad = displacement.clone().requires_grad_()
    function(echo_for_grad, displacement_for_grad).sum().backward()

    for value in (
        output,
        tangent,
        echo_gradient,
        displacement_gradient,
        echo_for_grad.grad,
        displacement_for_grad.grad,
    ):
        torch.testing.assert_close(value, torch.zeros_like(value))
        assert bool(torch.all(torch.isfinite(value)))


def test_cross_dtype_integer_boundary_keeps_frozen_branch_derivative() -> None:
    echo = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    displacement = torch.tensor((1.0, 0.0), dtype=torch.float64)
    cell = RemapCell(1, 0)

    output, tangent = torch.func.jvp(
        lambda motion: remap_core(echo, motion, cell),
        (displacement,),
        (torch.ones_like(displacement),),
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tangent,
        torch.tensor([[0.0, 0.0], [-2.0, -3.0]], dtype=torch.float32),
    )
    assert bool(torch.all(torch.isfinite(tangent)))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS device is unavailable",
)
def test_mps_float32_echo_accepts_cpu_float64_displacement() -> None:
    echo = torch.arange(4, dtype=torch.float32, device="mps").reshape(2, 2)
    displacement = torch.tensor((1.0, 0.0), dtype=torch.float64)

    output, tangent = torch.func.jvp(
        lambda field, motion: remap(field, motion),
        (echo, displacement),
        (torch.ones_like(echo), torch.ones_like(displacement)),
    )

    torch.testing.assert_close(
        output.cpu(),
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tangent.cpu(),
        torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32),
    )
    assert output.device.type == "mps"
    assert tangent.device.type == "mps"
    assert bool(torch.all(torch.isfinite(tangent)))


def _linearization_fixture() -> tuple[torch.Tensor, variational.AnalysisLinearization]:
    frames = torch.stack(
        [torch.full((3, 3), value, dtype=torch.float64) for value in (20.0, 19.0, 18.0)]
    )
    config = variational.AnalysisConfig(
        censored_background_policy="detection_limit",
        maximum_outer_iterations=8,
        maximum_pcg_iterations=100,
        pcg_relative_tolerance=1.0e-8,
    )
    observations, frozen = variational.prepare_analysis(
        frames,
        analysis_config=config,
    )
    result = variational.solve_analysis(observations, frozen)
    assert result.linearization is not None
    return result.control, result.linearization


@pytest.mark.parametrize("kind", ("wrong_dtype", "extra_tail", "short"))
def test_linearization_admission_rejects_malformed_control(kind: str) -> None:
    control, linearization = _linearization_fixture()
    if kind == "wrong_dtype":
        candidate = control.float()
    elif kind == "extra_tail":
        candidate = torch.cat((control, torch.zeros(1, dtype=control.dtype)))
    else:
        candidate = control[:-1]
    addressed = variational._content_address_linearization(
        candidate,
        linearization,
    )

    with pytest.raises(ValueError, match="control"):
        variational.validate_analysis_linearization_content(
            candidate,
            addressed,
            require_current_environment=False,
        )
