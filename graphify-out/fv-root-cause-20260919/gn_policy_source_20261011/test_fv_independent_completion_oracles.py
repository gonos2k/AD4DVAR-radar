"""Independent multi-lead and adjoint completion oracles.

These are synthetic numerical checks.  The first uses a continuous
characteristic solution, rather than another finite-volume time integrator;
the second checks the implicit adjoint against the known derivative of its
manufactured solution.
"""

import math

import pytest
import torch
from torch.func import jacrev

from advar.transport import face_volume_fluxes, finite_volume_step


DTYPE = torch.float64


def _characteristic_cell_means(n, t, velocity, growth, *, device="cpu"):
    """Exact means for an affine initial field on [0,1]^2."""
    dx = 1.0 / n
    y, x = torch.meshgrid(
        (torch.arange(n, dtype=DTYPE, device=device) + 0.5) * dx,
        (torch.arange(n, dtype=DTYPE, device=device) + 0.5) * dx,
        indexing="ij",
    )
    u, v = velocity
    return math.exp(growth * t) * (2.0 + 0.1 * (x - u * t) + 0.08 * (y - v * t))


def _characteristic_face_means(n, t, velocity, growth):
    """Exact whole-face means of the same affine characteristic field."""
    dx = 1.0 / n
    centers = (torch.arange(n, dtype=DTYPE) + 0.5) * dx
    u, v = velocity
    scale = math.exp(growth * t)

    def q(x, y):
        return scale * (2.0 + 0.1 * (x - u * t) + 0.08 * (y - v * t))

    return q(torch.zeros_like(centers), centers), q(torch.ones_like(centers), centers), \
        q(centers, torch.zeros_like(centers)), q(centers, torch.ones_like(centers))


def _advance_characteristics(n, lead, velocity, growth):
    dx, dy = 1.0 / n, 1.0 / n
    u, v = velocity
    vertices = torch.arange(n + 1, dtype=DTYPE) * dx
    yv, xv = torch.meshgrid(vertices, vertices, indexing="ij")
    # Linear streamfunction gives constant (u, v) and exactly zero divergence.
    qx, qy = face_volume_fluxes(u * yv - v * xv)
    echo = _characteristic_cell_means(n, 0.0, velocity, growth)
    support = torch.ones_like(echo)
    dt_limit = 0.4 / (abs(u) / dx + abs(v) / dy)
    count = math.ceil(lead / dt_limit)
    dt = lead / count
    budget = 0.0
    for step in range(count):
        t0 = step * dt
        edges = (_characteristic_face_means(n, t0, velocity, growth),
                 _characteristic_face_means(n, t0 + dt, velocity, growth))
        result = finite_volume_step(
            echo, support, qx, qy, dt_seconds=dt, spacing_yx=(dy, dx),
            log_growth=growth * dt, boundary_echo=edges,
            boundary_support=((torch.ones(n, dtype=DTYPE),) * 4,) * 2,
            reconstruction="minmod",
        )
        assert float(result.echo.min()) >= 0.0
        assert float(result.support.min()) >= 0.0
        assert float(result.support.max()) <= 1.0
        budget += float(result.physical_inflow - result.physical_outflow + result.source_quadrature)
        echo, support = result.echo, result.support
    return echo, budget


def test_transport_characteristic_oracle_covers_multiple_future_leads_and_refinement():
    velocity, growth = (0.12, -0.04), 0.08
    errors = []
    for n in (8, 16):
        per_lead = []
        for lead in (0.4, 0.8):
            actual, budget = _advance_characteristics(n, lead, velocity, growth)
            truth = _characteristic_cell_means(n, lead, velocity, growth)
            rel = torch.linalg.vector_norm(actual - truth) / torch.linalg.vector_norm(truth)
            per_lead.append(float(rel))
            # This is the echo-proxy integral, not water mass. The physical-time
            # flux/source quadrature has truncation error, unlike exact budget identities.
            initial = _characteristic_cell_means(n, 0.0, velocity, growth).sum() / (n * n)
            final = actual.sum() / (n * n)
            dt_limit = 0.4 / (n * (abs(velocity[0]) + abs(velocity[1])))
            dt = lead / math.ceil(lead / dt_limit)
            advective_scale = 0.1 * abs(velocity[0]) + 0.08 * abs(velocity[1])
            # A conservative second-order scale for physical-time growth/flux
            # quadrature, with q/time derivatives from this manufactured field.
            quadrature_scale = abs(growth)**3 * float(truth.mean()) + growth**2 * advective_scale
            tolerance = quadrature_scale * lead * dt**2 + 256 * torch.finfo(DTYPE).eps * float(initial)
            assert abs(float(final - initial - budget)) <= tolerance
            assert float(actual.min()) > 0.0
        errors.append(per_lead[-1])
        assert per_lead[-1] < 1e-3
    # Refine spacing and the CFL-limited time step together. This demonstrates
    # decreasing error, not an asymptotic order or operational forecast skill.
    assert errors[1] < 0.6 * errors[0]


def _manufactured_adjoint(dtype=DTYPE):
    kappa, beta = 0.7, 0.13
    p = torch.tensor(0.23, dtype=dtype)
    xbase = torch.linspace(0.1, 0.4, 25, dtype=dtype)
    v = torch.linspace(-0.2, 0.3, 25, dtype=dtype)
    w = torch.linspace(0.1, -0.15, 25, dtype=dtype)
    K = torch.linspace(0.8, 1.4, 25, dtype=dtype)
    M = torch.arange(1, 53, dtype=dtype).reshape(2, 26) / 31.0
    target = torch.tensor([0.4, -0.2], dtype=dtype)

    def optimum(parameter):
        xstar = xbase + parameter * v + 0.5 * parameter.square() * w
        ystar = 0.5 * kappa * xstar[0].square()
        return torch.cat((xstar, ystar.reshape(1)))

    def fields(c, parameter):
        q = c[25] - 0.5 * kappa * c[0].square()
        delta = c[:25] - (xbase + parameter * v + 0.5 * parameter.square() * w)
        base = 0.5 * (K * delta.square()).sum() + 0.5 * q.square() + 0.1 * q
        return base - 0.4 * q, base + 0.6 * q, q

    def objective(c, parameter):
        residual = M @ c - target
        return 0.5 * residual.square().sum() + beta * parameter

    def F(z, parameter):
        c, theta = z[:26], z[26]
        minus, plus, q = fields(c, parameter)
        grad_minus = jacrev(lambda value: fields(value, parameter)[0])(c)
        grad_plus = jacrev(lambda value: fields(value, parameter)[1])(c)
        return torch.cat(((1.0 - theta) * grad_minus + theta * grad_plus,
                          (q / 0.84).reshape(1)))

    root = torch.cat((optimum(p), torch.tensor([0.3], dtype=dtype)))
    return p, v, w, kappa, beta, M, target, optimum, objective, F, root


def test_adjoint_synthetic_future_score_matches_direct_and_reanalysis_derivatives():
    p, v, w, kappa, beta, M, target, optimum, objective, F, root = _manufactured_adjoint()
    Fz = jacrev(F, argnums=0)(root, p)
    Fp = jacrev(F, argnums=1)(root, p)
    c = root[:26]
    Ec = jacrev(lambda value: objective(value, p))(c)
    Ep = jacrev(lambda value: objective(c, value))(p)
    rhs = torch.cat((Ec, Ec.new_zeros(1)))
    lam = torch.linalg.solve(Fz.mT, rhs)
    adjoint_total = Ep - torch.dot(lam, Fp)
    direct_zp = torch.linalg.solve(Fz, -Fp)

    cprime_x = v + p * w
    cprime_y = kappa * optimum(p)[0] * cprime_x[0]
    cprime = torch.cat((cprime_x, cprime_y.reshape(1)))
    analytic = torch.dot(M @ c - target, M @ cprime) + beta
    torch.testing.assert_close(F(root, p), torch.zeros_like(root), atol=2e-12, rtol=0)
    torch.testing.assert_close(direct_zp[:26], cprime, atol=2e-11, rtol=0)
    torch.testing.assert_close(adjoint_total, analytic, atol=2e-11, rtol=0)
    torch.testing.assert_close(Fz.mT @ lam, rhs, atol=2e-11, rtol=0)

    central_errors = []
    for h in (2e-3, 1e-3):
        plus = objective(optimum(p + h), p + h)
        minus = objective(optimum(p - h), p - h)
        central = (plus - minus) / (2 * h)
        central_errors.append(float(abs(central - analytic)))
        assert central_errors[-1] < 2e-5
    assert central_errors[1] < .3 * central_errors[0]
