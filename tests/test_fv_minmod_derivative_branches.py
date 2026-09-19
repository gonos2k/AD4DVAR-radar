"""Small public-transport checks for minmod's piecewise derivative.

These tests deliberately separate an AD-selected
linearization from a two-sided finite-difference derivative at a limiter tie.
They exercise ``finite_volume_step`` directly and make no claim about an
inverse response or a differentiability certificate for minmod.
"""

import torch

from advar.transport import finite_volume_step


DTYPE = torch.float64
HEIGHT, WIDTH = 5, 9


def _boundary(value: float, *, dtype: torch.dtype = DTYPE):
    edge = (
        torch.full((HEIGHT,), value, dtype=dtype),
        torch.full((HEIGHT,), value, dtype=dtype),
        torch.full((WIDTH,), value, dtype=dtype),
        torch.full((WIDTH,), value, dtype=dtype),
    )
    return (edge, edge)


def _minmod_x_operator(echo: torch.Tensor, qx: torch.Tensor, qy: torch.Tensor):
    support = torch.ones_like(echo)
    return finite_volume_step(
        echo,
        support,
        qx,
        qy,
        dt_seconds=0.2,
        spacing_yx=(1.0, 1.0),
        boundary_echo=_boundary(0.0, dtype=echo.dtype),
        boundary_support=_boundary(1.0, dtype=echo.dtype),
        max_courant=0.2,
        reconstruction="minmod",
    ).echo


def _x_flow() -> tuple[torch.Tensor, torch.Tensor]:
    # Constant positive qx gives a divergence-free x flow with CFL .2.
    return (
        torch.ones((HEIGHT, WIDTH + 1), dtype=DTYPE),
        torch.zeros((HEIGHT + 1, WIDTH), dtype=DTYPE),
    )


def test_equal_slope_limiter_has_one_sided_split_even_when_ad_vjp_is_consistent():
    echo = torch.arange(1, WIDTH + 1, dtype=DTYPE).repeat(HEIGHT, 1).requires_grad_()
    qx, qy = _x_flow()
    direction = torch.zeros_like(echo)
    direction[2, 4] = 1.0

    def forward(value: torch.Tensor) -> torch.Tensor:
        return _minmod_x_operator(value, qx, qy)

    base, jvp = torch.func.jvp(forward, (echo,), (direction,))
    # The equal neighboring slopes are a minmod selector tie.  The AD result
    # remains an AD-selected linearization, while the two directional limits
    # are different; it need not be the derivative of either smooth branch.
    limits = []
    for eps in (1e-4, 1e-5, 1e-6):
        plus = (forward(echo.detach() + eps * direction) - base.detach()) / eps
        minus = (base.detach() - forward(echo.detach() - eps * direction)) / eps
        assert float(torch.linalg.vector_norm(plus - minus).detach()) > 1e-2
        assert float(torch.linalg.vector_norm(jvp - plus).detach()) > 1e-2
        assert float(torch.linalg.vector_norm(jvp - minus).detach()) > 1e-2
        limits.append(torch.stack((plus, minus)))
    torch.testing.assert_close(limits[-1], limits[0], atol=5e-8, rtol=5e-8)

    cotangent = torch.linspace(0.2, 1.1, echo.numel(), dtype=DTYPE).reshape_as(echo)
    (vjp,) = torch.autograd.grad((forward(echo) * cotangent).sum(), (echo,))
    torch.testing.assert_close(
        (jvp * cotangent).sum(),
        (direction * vjp).sum(),
        atol=2e-12,
        rtol=2e-12,
    )


def test_smooth_quadratic_profile_has_matching_one_sided_finite_differences():
    rows, columns = torch.meshgrid(
        torch.arange(HEIGHT, dtype=DTYPE),
        torch.arange(WIDTH, dtype=DTYPE),
        indexing="ij",
    )
    # Positive, monotone x slopes avoid the zero-slope and equal-selector
    # boundaries while retaining a genuinely quadratic profile.
    echo = (1.0 + 0.02 * (columns + 1).square() + 0.01 * rows).requires_grad_()
    direction = 0.01 * (columns + 1) + 0.002 * rows
    qx, qy = _x_flow()

    def forward(value: torch.Tensor) -> torch.Tensor:
        return _minmod_x_operator(value, qx, qy)

    base, jvp = torch.func.jvp(forward, (echo,), (direction,))
    eps = 1e-5
    plus = (forward(echo.detach() + eps * direction) - base.detach()) / eps
    minus = (base.detach() - forward(echo.detach() - eps * direction)) / eps
    torch.testing.assert_close(jvp, plus, atol=5e-9, rtol=5e-8)
    torch.testing.assert_close(jvp, minus, atol=5e-9, rtol=5e-8)


def test_zero_flow_at_limiter_ties_is_identity_with_identity_derivative():
    echo = torch.arange(1, WIDTH + 1, dtype=DTYPE).repeat(HEIGHT, 1)
    direction = torch.sin(torch.arange(echo.numel(), dtype=DTYPE).reshape_as(echo))
    qx = torch.zeros((HEIGHT, WIDTH + 1), dtype=DTYPE)
    qy = torch.zeros((HEIGHT + 1, WIDTH), dtype=DTYPE)

    def forward(value: torch.Tensor) -> torch.Tensor:
        return _minmod_x_operator(value, qx, qy)

    base, jvp = torch.func.jvp(forward, (echo,), (direction,))
    torch.testing.assert_close(base, echo, atol=0.0, rtol=0.0)
    torch.testing.assert_close(jvp, direction, atol=0.0, rtol=0.0)
