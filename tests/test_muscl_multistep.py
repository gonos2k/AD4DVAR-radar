"""Independent three-step AD and conservation checks for the MUSCL experiment.

The limiter makes the map only piecewise smooth.  These tests compare a real
three-step composition with an independent finite-volume oracle, and label a
Taylor probe when its two perturbed trajectories use different limiter
branches.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


_SPEC = spec_from_file_location(
    "muscl_experiment_multistep_under_test",
    Path(__file__).parents[1] / "examples" / "weather_scenarios" / "muscl_experiment.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
muscl_step = _MODULE.muscl_step


def _independent_minmod(left, right):
    same_positive = (left > 0) & (right > 0)
    same_negative = (left < 0) & (right < 0)
    return torch.where(
        same_positive,
        torch.minimum(left, right),
        torch.where(same_negative, torch.maximum(left, right), torch.zeros_like(left)),
    )


def _independent_faces(q):
    height, width = q.shape
    sx = torch.zeros_like(q)
    sy = torch.zeros_like(q)
    if height > 2 and width > 2:
        sx[1:-1, 1:-1] = _independent_minmod(
            q[1:-1, 1:-1] - q[1:-1, :-2],
            q[1:-1, 2:] - q[1:-1, 1:-1],
        )
        sy[1:-1, 1:-1] = _independent_minmod(
            q[1:-1, 1:-1] - q[:-2, 1:-1],
            q[2:, 1:-1] - q[1:-1, 1:-1],
        )
    return q - sx / 2, q + sx / 2, q - sy / 2, q + sy / 2


def _independent_euler(q, qx, qy, dt, area):
    left, right, bottom, top = _independent_faces(q)
    xp, xm = torch.clamp_min(qx, 0), torch.clamp_max(qx, 0)
    yp, ym = torch.clamp_min(qy, 0), torch.clamp_max(qy, 0)
    zero_x, zero_y = torch.zeros_like(q[:, :1]), torch.zeros_like(q[:1, :])
    incoming = (
        xp[:, :-1] * torch.cat((zero_x, right[:, :-1]), dim=1)
        - xm[:, 1:] * torch.cat((left[:, 1:], zero_x), dim=1)
        + yp[:-1, :] * torch.cat((zero_y, top[:-1, :]), dim=0)
        - ym[1:, :] * torch.cat((bottom[1:, :], zero_y), dim=0)
    )
    outgoing = xp[:, 1:] * right - xm[:, :-1] * left + yp[1:, :] * top - ym[:-1, :] * bottom
    boundary = (
        (-xm[:, 0] * left[:, 0]).sum()
        + (xp[:, -1] * right[:, -1]).sum()
        + (-ym[0, :] * bottom[0, :]).sum()
        + (yp[-1, :] * top[-1, :]).sum()
    )
    return q + dt * (incoming - outgoing) / area, boundary


def _independent_step(q, qx, qy, dt_seconds, spacing_yx, log_growth):
    area = q.new_tensor(spacing_yx[0] * spacing_yx[1])
    dt = q.new_tensor(dt_seconds)
    stage1, out0 = _independent_euler(q, qx, qy, dt, area)
    stage2, out1 = _independent_euler(stage1, qx, qy, dt, area)
    growth = torch.exp(q.new_tensor(log_growth))
    return growth * (q + stage2) / 2, dt * (out0 + out1) / 2


def _case():
    dtype = torch.float64
    height, width = 6, 7
    vertex_y, vertex_x = torch.meshgrid(
        torch.linspace(-1, 1, height + 1, dtype=dtype),
        torch.linspace(-1, 1, width + 1, dtype=dtype),
        indexing="ij",
    )
    # Nonuniform streamfunction: differencing it gives exact telescoping
    # discrete divergence zero while retaining mixed-sign face fluxes.
    psi = (
        0.10 * vertex_y.square() * vertex_x
        + 0.07 * torch.sin(0.4 * vertex_y) * torch.cos(0.5 * vertex_x)
        + 0.03 * vertex_y * vertex_x.square()
        + 0.02 * vertex_y
        - 0.015 * vertex_x
    )
    cell_y, cell_x = torch.meshgrid(
        torch.arange(height, dtype=dtype), torch.arange(width, dtype=dtype), indexing="ij"
    )
    q = 0.75 + torch.exp(0.10 * cell_y + 0.06 * cell_x) + 0.03 * torch.sin(
        0.7 * cell_y + 0.41 * cell_x
    )
    qx, qy = psi[1:, :] - psi[:-1, :], -(psi[:, 1:] - psi[:, :-1])
    return q, psi, qx, qy, 0.35, (1.3, 0.8), 0.11


def _compose_actual(q, qx, qy, dt, spacing, log_growth):
    budgets = []
    for _ in range(3):
        q, budget = muscl_step(
            q, qx, qy, dt_seconds=dt, spacing_yx=spacing, log_growth=log_growth
        )
        budgets.append(budget)
    return q, tuple(budgets)


def _compose_independent(q, qx, qy, dt, spacing, log_growth):
    budgets = []
    for _ in range(3):
        q, budget = _independent_step(q, qx, qy, dt, spacing, log_growth)
        budgets.append(budget)
    return q, tuple(budgets)


def _branch_code(q):
    def one_axis(left, center, right):
        dl, dr = center - left, right - center
        code = torch.zeros_like(dl, dtype=torch.int8)
        positive = (dl > 0) & (dr > 0)
        negative = (dl < 0) & (dr < 0)
        # The sign branch alone is insufficient: minmod also chooses the
        # smaller magnitude, and that selector can cross during refinement.
        code = torch.where(positive & (dl <= dr), torch.ones_like(code), code)
        code = torch.where(positive & (dl > dr), 2 * torch.ones_like(code), code)
        code = torch.where(negative & (dl >= dr), -torch.ones_like(code), code)
        return torch.where(negative & (dl < dr), -2 * torch.ones_like(code), code)

    x_code = one_axis(q[:, :-2], q[:, 1:-1], q[:, 2:])
    y_code = one_axis(q[:-2, :], q[1:-1, :], q[2:, :])
    return tuple(x_code.reshape(-1).tolist() + y_code.reshape(-1).tolist())


def _trajectory_branches(q, qx, qy, dt, spacing, log_growth):
    area = q.new_tensor(spacing[0] * spacing[1])
    dt_tensor = q.new_tensor(dt)
    growth = torch.exp(q.new_tensor(log_growth))
    branches = [tuple(qx.sign().flatten().tolist()), tuple(qy.sign().flatten().tolist())]
    for _ in range(3):
        branches.append(_branch_code(q))
        stage1, _ = _independent_euler(q, qx, qy, dt_tensor, area)
        branches.append(_branch_code(stage1))
        stage2, _ = _independent_euler(stage1, qx, qy, dt_tensor, area)
        branches.append(_branch_code(stage2))
        q = growth * (q + stage2) / 2
    return tuple(branches)


def test_three_step_composition_matches_independent_oracle_and_weighted_budget():
    q, _, qx, qy, dt, spacing, log_growth = _case()
    assert bool((q > 0).all()) and float(q.std()) > 0.0
    divergence = qx[:, 1:] - qx[:, :-1] + qy[1:, :] - qy[:-1, :]
    torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-14, rtol=0)
    assert float(qx.min()) < 0.0 < float(qx.max())
    assert float(qy.min()) < 0.0 < float(qy.max())
    area = spacing[0] * spacing[1]
    outgoing_rate = (
        torch.clamp_min(qx[:, 1:], 0)
        - torch.clamp_max(qx[:, :-1], 0)
        + torch.clamp_min(qy[1:, :], 0)
        - torch.clamp_max(qy[:-1, :], 0)
    ) / area
    assert float(dt * outgoing_rate.max()) < 0.5
    actual, actual_budget = _compose_actual(q, qx, qy, dt, spacing, log_growth)
    expected, expected_budget = _compose_independent(q, qx, qy, dt, spacing, log_growth)
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-13)
    for observed, reference in zip(actual_budget, expected_budget):
        torch.testing.assert_close(observed, reference, atol=3e-13, rtol=3e-13)

    lhs = torch.exp(q.new_tensor(-3 * log_growth)) * actual.sum() * area
    rhs = q.sum() * area - actual_budget[0] - torch.exp(q.new_tensor(-log_growth)) * actual_budget[1]
    rhs = rhs - torch.exp(q.new_tensor(-2 * log_growth)) * actual_budget[2]
    torch.testing.assert_close(lhs, rhs, atol=5e-12, rtol=5e-12)


def test_three_step_q_jvp_vjp_match_actual_central_difference():
    q, _, qx, qy, dt, spacing, log_growth = _case()
    q = q.requires_grad_()
    index = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)
    direction = 0.025 * torch.sin(0.31 * index) + 0.011 * torch.cos(0.17 * index.square())
    cotangent = 0.2 + 0.03 * torch.sin(0.23 * index + 0.4)

    def forward(value):
        return _compose_actual(value, qx, qy, dt, spacing, log_growth)[0]

    _, jvp = torch.func.jvp(forward, (q,), (direction,))
    eps = 1e-6
    finite_difference = (forward(q + eps * direction) - forward(q - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite_difference, atol=5e-9, rtol=5e-8)
    (vjp,) = torch.autograd.grad((forward(q) * cotangent).sum(), (q,))
    torch.testing.assert_close(
        (jvp * cotangent).sum(), (direction * vjp).sum(), atol=5e-9, rtol=5e-8
    )


def test_three_step_psi_jvp_vjp_match_actual_central_difference():
    q, psi, _, _, dt, spacing, log_growth = _case()
    psi = psi.requires_grad_()
    vertex_index = torch.arange(psi.numel(), dtype=psi.dtype).reshape_as(psi)
    direction = 0.017 * torch.sin(0.37 * vertex_index) + 0.013 * torch.cos(0.19 * vertex_index.square())
    cotangent = 0.4 + 0.05 * torch.sin(torch.arange(q.numel(), dtype=q.dtype).reshape_as(q))

    def forward(value):
        qx = value[1:, :] - value[:-1, :]
        qy = -(value[:, 1:] - value[:, :-1])
        return _compose_actual(q, qx, qy, dt, spacing, log_growth)[0]

    _, jvp = torch.func.jvp(forward, (psi,), (direction,))
    eps = 1e-6
    finite_difference = (forward(psi + eps * direction) - forward(psi - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite_difference, atol=5e-9, rtol=5e-8)
    (vjp,) = torch.autograd.grad((forward(psi) * cotangent).sum(), (psi,))
    torch.testing.assert_close(
        (jvp * cotangent).sum(), (direction * vjp).sum(), atol=5e-9, rtol=5e-8
    )


def test_three_step_growth_jvp_vjp_match_actual_central_difference():
    q, _, qx, qy, dt, spacing, log_growth = _case()
    growth = q.new_tensor(log_growth).requires_grad_()
    direction = q.new_tensor(1.0)
    cotangent = 0.2 + 0.03 * torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)

    def forward(value):
        return _compose_actual(q, qx, qy, dt, spacing, value)[0]

    _, jvp = torch.func.jvp(forward, (growth,), (direction,))
    eps = 1e-6
    finite_difference = (forward(growth + eps) - forward(growth - eps)) / (2 * eps)
    torch.testing.assert_close(jvp, finite_difference, atol=5e-9, rtol=5e-8)
    (vjp,) = torch.autograd.grad((forward(growth) * cotangent).sum(), (growth,))
    torch.testing.assert_close(
        (jvp * cotangent).sum(), direction * vjp, atol=5e-9, rtol=5e-8
    )


def test_zero_growth_three_step_mass_budget_residual_has_zero_derivative():
    q, _, qx, qy, dt, spacing, _ = _case()
    area = spacing[0] * spacing[1]
    q = q.requires_grad_()
    index = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)
    direction = 0.02 * torch.sin(0.29 * index) + 0.009 * torch.cos(0.13 * index.square())

    def residual(value):
        final, budgets = _compose_actual(value, qx, qy, dt, spacing, 0.0)
        return final.sum() * area + sum(budgets) - value.sum() * area

    _, jvp = torch.func.jvp(residual, (q,), (direction,))
    # At 1e-6 the subtraction of two nearly equal mass residuals loses a few
    # ulps; this scale remains in the centered-FD asymptotic regime.
    eps = 1e-4
    finite_difference = (residual(q + eps * direction) - residual(q - eps * direction)) / (2 * eps)
    torch.testing.assert_close(jvp, finite_difference, atol=5e-9, rtol=5e-8)
    (vjp,) = torch.autograd.grad(residual(q), (q,))
    assert float(torch.linalg.vector_norm(vjp)) < 5e-11
    assert abs(jvp.detach().item()) < 5e-11


def test_taylor_probe_classifies_limiter_crossing_before_claiming_refinement():
    q, psi, qx, qy, dt, spacing, log_growth = _case()
    index = torch.arange(q.numel(), dtype=q.dtype).reshape_as(q)
    # This q direction deliberately crosses a limiter selector at coarse h.
    q_direction = 0.1 * (
        torch.sin(0.43 * index + 0.27) + 0.25 * torch.cos(0.71 * index - 0.19)
    )
    scales = (0.20, 0.10, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003)
    baseline = _trajectory_branches(q, qx, qy, dt, spacing, log_growth)
    q_classifications = []
    for scale in scales:
        plus = _trajectory_branches(q + scale * q_direction, qx, qy, dt, spacing, log_growth)
        minus = _trajectory_branches(q - scale * q_direction, qx, qy, dt, spacing, log_growth)
        q_classifications.append(
            "branch-crossing-no-Taylor-claim"
            if plus != baseline or minus != baseline
            else "sampled-fixed-branch"
        )
    assert q_classifications[0] == "branch-crossing-no-Taylor-claim"
    assert q_classifications[-1] == "sampled-fixed-branch"

    # A streamfunction perturbation changes the flux and therefore gives a
    # genuinely nonlinear composition.  Refine only entries sampled on the
    # same limiter trajectory; a crossing is explicitly excluded from the
    # Taylor claim.
    vertex_index = torch.arange(psi.numel(), dtype=psi.dtype).reshape_as(psi)
    psi_direction = 0.1 * (
        torch.sin(0.37 * vertex_index)
        + 0.4 * torch.cos(0.19 * vertex_index.square())
    )
    psi_baseline = _trajectory_branches(q, qx, qy, dt, spacing, log_growth)
    psi_classifications = []
    errors = []

    def forward(value):
        value_qx = value[1:, :] - value[:-1, :]
        value_qy = -(value[:, 1:] - value[:, :-1])
        return _compose_actual(q, value_qx, value_qy, dt, spacing, log_growth)[0]

    _, jvp = torch.func.jvp(forward, (psi,), (psi_direction,))
    for scale in scales:
        plus_psi = psi + scale * psi_direction
        minus_psi = psi - scale * psi_direction
        plus_qx = plus_psi[1:, :] - plus_psi[:-1, :]
        plus_qy = -(plus_psi[:, 1:] - plus_psi[:, :-1])
        minus_qx = minus_psi[1:, :] - minus_psi[:-1, :]
        minus_qy = -(minus_psi[:, 1:] - minus_psi[:, :-1])
        plus = _trajectory_branches(q, plus_qx, plus_qy, dt, spacing, log_growth)
        minus = _trajectory_branches(q, minus_qx, minus_qy, dt, spacing, log_growth)
        if plus != psi_baseline or minus != psi_baseline:
            psi_classifications.append("branch-crossing-no-Taylor-claim")
            continue
        psi_classifications.append("sampled-fixed-branch")
        remainder = forward(psi + scale * psi_direction) - forward(psi) - scale * jvp
        errors.append(float(torch.linalg.vector_norm(remainder)))

    assert psi_classifications[0] == "branch-crossing-no-Taylor-claim"
    assert psi_classifications[-1] == "sampled-fixed-branch"
    assert "branch-crossing-no-Taylor-claim" in psi_classifications
    assert "sampled-fixed-branch" in psi_classifications
    assert len(errors) >= 2
    # Only sampled-fixed-branch entries participate; require actual refinement
    # evidence rather than passing when every useful sample was excluded.
    assert errors[-1] < errors[0] * 0.02
