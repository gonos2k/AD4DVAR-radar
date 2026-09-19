"""Independent checks for the standalone minmod-MUSCL experiment.

The oracle here deliberately reimplements the reconstruction and SSPRK2
update.  The experiment has a narrow contract: full known ``q``, homogeneous
zero exterior, a stationary divergence-free face flow, and a scalar
integrating factor.  These tests do not certify the production transport or
any FSO/learning decomposition.
"""

import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

_MODULE_SPEC = spec_from_file_location(
    "muscl_experiment_under_test",
    Path(__file__).parents[1] / "examples" / "weather_scenarios" / "muscl_experiment.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
muscl_step = _MODULE.muscl_step


def _minmod(left, right):
    same_sign = ((left > 0) & (right > 0)) | ((left < 0) & (right < 0))
    return torch.where(
        same_sign,
        torch.where(torch.abs(left) <= torch.abs(right), left, right),
        torch.zeros_like(left),
    )


def _oracle_faces(q):
    """Return independently reconstructed cell faces with zero edge slopes."""
    h, w = q.shape
    sx = torch.zeros_like(q)
    sy = torch.zeros_like(q)
    if w > 2:
        sx[1:-1, 1:-1] = _minmod(
            q[1:-1, 1:-1] - q[1:-1, :-2], q[1:-1, 2:] - q[1:-1, 1:-1]
        )
    if h > 2:
        sy[1:-1, 1:-1] = _minmod(
            q[1:-1, 1:-1] - q[:-2, 1:-1], q[2:, 1:-1] - q[1:-1, 1:-1]
        )
    return q - 0.5 * sx, q + 0.5 * sx, q - 0.5 * sy, q + 0.5 * sy


def _oracle_euler(q, qx, qy, dt, area):
    left, right, bottom, top = _oracle_faces(q)
    qx_plus = torch.clamp_min(qx, 0)
    qx_minus = torch.clamp_max(qx, 0)
    qy_plus = torch.clamp_min(qy, 0)
    qy_minus = torch.clamp_max(qy, 0)
    zero_x = torch.zeros_like(q[:, :1])
    zero_y = torch.zeros_like(q[:1, :])
    incoming = (
        qx_plus[:, :-1] * torch.cat((zero_x, right[:, :-1]), dim=1)
        - qx_minus[:, 1:] * torch.cat((left[:, 1:], zero_x), dim=1)
        + qy_plus[:-1, :] * torch.cat((zero_y, top[:-1, :]), dim=0)
        - qy_minus[1:, :] * torch.cat((bottom[1:, :], zero_y), dim=0)
    )
    outgoing = (
        qx_plus[:, 1:] * right
        - qx_minus[:, :-1] * left
        + qy_plus[1:, :] * top
        - qy_minus[:-1, :] * bottom
    )
    boundary_out = (
        (-qx_minus[:, 0] * left[:, 0]).sum()
        + (qx_plus[:, -1] * right[:, -1]).sum()
        + (-qy_minus[0, :] * bottom[0, :]).sum()
        + (qy_plus[-1, :] * top[-1, :]).sum()
    )
    return q + dt * (incoming - outgoing) / area, boundary_out


def _oracle_step(q, qx, qy, dt, spacing, log_growth=0.0):
    area = q.new_tensor(float(spacing[0]) * float(spacing[1]))
    r1, out0 = _oracle_euler(q, qx, qy, q.new_tensor(dt), area)
    r2, out1 = _oracle_euler(r1, qx, qy, q.new_tensor(dt), area)
    growth = torch.exp(q.new_tensor(log_growth))
    return growth * 0.5 * (q + r2), q.new_tensor(dt) * 0.5 * (out0 + out1)


def _constant_flow(h, w, *, x=0.0, y=0.0, dtype=torch.float64):
    return (
        torch.full((h, w + 1), x, dtype=dtype),
        torch.full((h + 1, w), y, dtype=dtype),
    )


def _psi_flow(psi):
    return psi[1:, :] - psi[:-1, :], -(psi[:, 1:] - psi[:, :-1])


def _positive_field(dtype=torch.float64):
    y, x = torch.meshgrid(
        torch.arange(5, dtype=dtype), torch.arange(5, dtype=dtype), indexing="ij"
    )
    return 0.25 + torch.exp(0.13 * y + 0.071 * x)


def test_matches_independent_muscl_ssprk2_oracle_on_nonsmooth_field():
    q = torch.tensor(
        [[0.0, 0.4, 1.2, 0.3, 0.0], [0.2, 1.0, 2.0, 0.8, 0.1],
         [0.5, 1.6, 0.7, 2.1, 0.4], [0.1, 0.9, 1.7, 0.6, 0.0],
         [0.0, 0.3, 0.8, 0.2, 0.0]], dtype=torch.float64
    )
    qx, qy = _constant_flow(5, 5, x=0.37, y=-0.29)
    actual = muscl_step(q, qx, qy, dt_seconds=0.4, spacing_yx=(1.7, 0.6))
    expected = _oracle_step(q, qx, qy, 0.4, (1.7, 0.6))
    torch.testing.assert_close(actual[0], expected[0], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-13, rtol=1e-13)


def test_signed_nonsquare_face_fluxes_match_oracle():
    q = _positive_field()
    qx, qy = _constant_flow(5, 5, x=-0.41, y=0.23)
    actual = muscl_step(q, qx, qy, dt_seconds=0.35, spacing_yx=(2.0, 0.75))
    expected = _oracle_step(q, qx, qy, 0.35, (2.0, 0.75))
    torch.testing.assert_close(actual[0], expected[0], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-13, rtol=1e-13)


def test_streamfunction_faces_are_divergence_free_and_match_oracle():
    y, x = torch.meshgrid(
        torch.arange(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64), indexing="ij"
    )
    psi = 0.31 * y - 0.47 * x + 0.02 * y.square() * x
    qx, qy = _psi_flow(psi)
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-14, rtol=0)
    q = _positive_field()
    actual = muscl_step(q, qx, qy, dt_seconds=0.02, spacing_yx=(1.3, 0.8))
    expected = _oracle_step(q, qx, qy, 0.02, (1.3, 0.8))
    torch.testing.assert_close(actual[0], expected[0], atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e-20), (torch.float32, 1.0), (torch.float32, 1e20), (torch.float64, 1e-200), (torch.float64, 1e200)])
def test_cfl_half_preserves_finite_nonnegative_echo_at_supported_scales(dtype, scale):
    q = scale * torch.tensor(
        [[0.0, 0.4, 1.0, 0.2, 0.0], [0.2, 1.0, 2.0, 0.8, 0.1],
         [0.5, 1.6, 0.7, 2.1, 0.4], [0.1, 0.9, 1.7, 0.6, 0.0],
         [0.0, 0.3, 0.8, 0.2, 0.0]], dtype=dtype
    )
    qx, qy = _constant_flow(5, 5, x=1.0, dtype=dtype)
    result, outflow = muscl_step(q, qx, qy, dt_seconds=0.5, spacing_yx=(1.0, 1.0))
    assert bool(torch.isfinite(result).all())
    assert bool((result >= 0).all())
    assert bool(torch.isfinite(outflow)) and float(outflow) >= 0


def test_cfl_above_half_is_rejected():
    q = _positive_field()
    qx, qy = _constant_flow(5, 5, x=1.0)
    with pytest.raises(ValueError, match="CFL|courant"):
        muscl_step(q, qx, qy, dt_seconds=0.5000001, spacing_yx=(1.0, 1.0))


@pytest.mark.parametrize("log_growth", [-0.4, 0.0, 0.35])
def test_no_flow_constant_integrating_factor_is_exact(log_growth):
    q = _positive_field()
    qx, qy = _constant_flow(5, 5)
    result, outflow = muscl_step(q, qx, qy, dt_seconds=0.2, spacing_yx=(2.0, 1.5), log_growth=log_growth)
    torch.testing.assert_close(result, q * math.exp(log_growth), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(outflow, torch.zeros_like(outflow), atol=0, rtol=0)


def test_nonzero_growth_mass_identity_uses_transformed_outflow():
    q = _positive_field()
    qx, qy = _constant_flow(5, 5, x=0.8, y=-0.35)
    dt = 0.2
    spacing = (1.6, 0.75)
    log_growth = 0.27
    result, transformed_outflow = muscl_step(
        q, qx, qy, dt_seconds=dt, spacing_yx=spacing, log_growth=log_growth
    )
    area = spacing[0] * spacing[1]
    lhs = math.exp(-log_growth) * float(result.sum()) * area
    rhs = float(q.sum()) * area - float(transformed_outflow)
    assert lhs == pytest.approx(rhs, rel=2e-13, abs=2e-13)


def test_zero_exterior_outflow_has_no_periodic_reentry():
    q = torch.zeros((5, 5), dtype=torch.float64)
    q[2, -1] = 1.0
    qx, qy = _constant_flow(5, 5, x=1.0)
    result, outflow = muscl_step(q, qx, qy, dt_seconds=0.5, spacing_yx=(1.0, 1.0))
    assert float(result[:, 0].sum()) == pytest.approx(0.0, abs=0.0)
    assert float(result[:, -1].sum()) > 0.0
    assert float(outflow) > 0.0


def test_q_jvp_vjp_and_finite_difference_agree_on_fixed_limiter_branches():
    q = _positive_field().requires_grad_()
    qx, qy = _constant_flow(5, 5, x=0.23, y=-0.17)
    direction = torch.linspace(-0.4, 0.5, q.numel(), dtype=q.dtype).reshape_as(q)
    cotangent = torch.linspace(0.2, 1.1, q.numel(), dtype=q.dtype).reshape_as(q)

    def forward(value):
        return muscl_step(value, qx, qy, dt_seconds=0.03, spacing_yx=(1.2, 0.9))[0]

    _, jvp = torch.func.jvp(forward, (q,), (direction,))
    eps = 1e-6
    finite = (forward(q + eps * direction) - forward(q - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite, atol=2e-8, rtol=2e-7)
    (vjp,) = torch.autograd.grad((forward(q) * cotangent).sum(), (q,))
    torch.testing.assert_close((jvp * cotangent).sum(), (direction * vjp).sum(), atol=2e-8, rtol=2e-7)


def test_psi_jvp_vjp_and_finite_difference_agree_away_from_flux_sign_changes():
    y, x = torch.meshgrid(
        torch.arange(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64), indexing="ij"
    )
    psi = (0.55 * y - 0.73 * x).requires_grad_()
    q = _positive_field()
    direction = .2 * torch.sin(torch.arange(psi.numel(), dtype=psi.dtype)).reshape_as(psi)
    cotangent = torch.linspace(0.1, 0.9, q.numel(), dtype=q.dtype).reshape_as(q)

    def forward(value):
        qx, qy = _psi_flow(value)
        return muscl_step(q, qx, qy, dt_seconds=0.02, spacing_yx=(1.1, 0.85))[0]

    _, jvp = torch.func.jvp(forward, (psi,), (direction,))
    eps = 1e-6
    finite = (forward(psi + eps * direction) - forward(psi - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite, atol=2e-8, rtol=2e-7)
    (vjp,) = torch.autograd.grad((forward(psi) * cotangent).sum(), (psi,))
    torch.testing.assert_close((jvp * cotangent).sum(), (direction * vjp).sum(), atol=2e-8, rtol=2e-7)


def test_minmod_known_field_is_explicitly_not_additive():
    # Embedded in a 5x5 field so the zero-slope perimeter contract is active.
    a = torch.zeros((5, 5), dtype=torch.float64)
    b = torch.zeros_like(a)
    a[2, 1:4] = torch.tensor([0.0, 1.0, 2.0])
    b[2, 1:4] = torch.tensor([0.0, 1.0, 0.0])
    _, a_right, _, _ = _oracle_faces(a)
    _, b_right, _, _ = _oracle_faces(b)
    _, total_right, _, _ = _oracle_faces(a + b)
    # At the center cell's right face: 2 != 1.5 + 1.0.  This is a documented
    # limiter property, so the test guards the counterexample rather than
    # imposing an invalid additive production contract.
    assert float(total_right[2, 2]) == pytest.approx(2.0)
    assert float(a_right[2, 2] + b_right[2, 2]) == pytest.approx(2.5)
    assert not torch.equal(total_right[2, 2], a_right[2, 2] + b_right[2, 2])
    qx, qy = _constant_flow(5, 5, x=1.)
    def advance(q):
        return muscl_step(q, qx, qy, dt_seconds=.1, spacing_yx=(1., 1.))[0]
    assert float((advance(a + b) - advance(a) - advance(b)).abs().max()) > .01
