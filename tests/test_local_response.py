"""Small analytic checks for the conditional matrix-free local response."""
from __future__ import annotations

import pytest
import torch

from advar.local_response import compute_local_response


def _quadratic_case():
    dtype = torch.float64
    A = torch.tensor([[3.0, 0.4, 0.1], [0.4, 2.0, -0.2], [0.1, -0.2, 1.5]], dtype=dtype)
    B = torch.tensor([[1.0, -0.3], [0.2, 0.8], [-0.4, 0.5]], dtype=dtype)
    C = torch.tensor([[0.7, -0.1, 0.2], [-0.3, 0.6, 0.4]], dtype=dtype)
    p = torch.tensor([0.4, -0.7], dtype=dtype)
    control = torch.linalg.solve(A, B @ p)

    def objective(c, q):
        return 0.5 * c @ A @ c - c @ (B @ q)

    def score(c, q):
        z = C @ c - q
        return 0.5 * z @ z + 0.2 * q @ q

    directions = {
        "p0": torch.tensor([1.0, -0.5], dtype=dtype),
        "p1": torch.tensor([-0.25, 0.75], dtype=dtype),
    }
    return A, B, C, control, p, objective, score, directions


def test_quadratic_response_has_direct_and_implicit_terms_without_dense_solver(monkeypatch):
    A, B, C, control, p, objective, score, directions = _quadratic_case()
    expected_adjoint = torch.linalg.solve(A, torch.func.grad(score, argnums=0)(control, p))
    expected_direct_gradient = torch.func.grad(score, argnums=1)(control, p)
    expected_indirect_gradient = B.T @ expected_adjoint
    expected_total_gradient = expected_direct_gradient + expected_indirect_gradient
    # This direction makes the total response cancel while both constituent
    # terms remain nonzero, so projections exercise the full gradient fields.
    directions = dict(directions)
    directions["cancellation"] = torch.stack(
        (expected_total_gradient[1], -expected_total_gradient[0])
    )
    monkeypatch.setattr(torch.linalg, "solve", lambda *args, **kwargs: pytest.fail("dense solve used"))
    result = compute_local_response(
        objective, score, control, p, directions,
        branch_check=lambda c, q: ("quadratic-branch", "fixed analytic branch"),
        input_identity={"case": "quadratic-v1", "dtype": "float64"},
    )
    expected_direct = torch.func.grad(score, argnums=1)(control, p)
    torch.testing.assert_close(result.direct_gradient, expected_direct_gradient, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(result.indirect_gradient, expected_indirect_gradient, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(result.total_gradient, expected_total_gradient, rtol=1e-11, atol=1e-12)
    for name, direction in directions.items():
        mixed = -B @ direction
        expected_indirect = -expected_adjoint @ mixed
        expected_total = expected_direct @ direction + expected_indirect
        torch.testing.assert_close(result.direct[name], expected_direct @ direction, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.indirect[name], expected_indirect, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.total[name], expected_total, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.direct[name], torch.dot(result.direct_gradient, direction), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.indirect[name], torch.dot(result.indirect_gradient, direction), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(result.total[name], torch.dot(result.total_gradient, direction), rtol=1e-11, atol=1e-12)
    assert result.gradient_max < 1e-10
    assert result.true_adjoint_residual < 1e-12
    assert result.pcg_relative_residual < 1e-10
    assert result.hvp_count > 0
    assert result.branch_signature == "quadratic-branch"
    assert result.scope.startswith("conditional on caller-supplied branch_check:")
    assert result.input_identity["case"] == "quadratic-v1"
    torch.testing.assert_close(result.score_control_gradient, C.T @ (C @ control - p), rtol=1e-11, atol=1e-12)
    for name, direction in directions.items():
        torch.testing.assert_close(result.mixed_gradients[name], -B @ direction, rtol=1e-11, atol=1e-12)
    assert abs(float(result.direct["cancellation"])) > 1e-3
    assert abs(float(result.indirect["cancellation"])) > 1e-3
    assert abs(float(result.total["cancellation"])) < 1e-12


def test_requires_float64_stationarity_branch_and_input_identity():
    _, _, _, control, p, objective, score, directions = _quadratic_case()
    kwargs = dict(
        branch_check=lambda c, q: ("branch", "scope"),
        input_identity={"case": "required"},
    )
    with pytest.raises(TypeError, match="float64"):
        compute_local_response(objective, score, control.float(), p.float(), directions, **kwargs)
    with pytest.raises(ValueError, match="stationary"):
        compute_local_response(objective, score, control + 1e-3, p, directions, **kwargs)
    with pytest.raises(ValueError, match="branch"):
        compute_local_response(objective, score, control, p, directions,
                               branch_check=lambda c, q: (_ for _ in ()).throw(RuntimeError("tie")),
                               input_identity={"case": "required"})
    with pytest.raises(TypeError, match="input_identity"):
        compute_local_response(objective, score, control, p, directions,
                               branch_check=lambda c, q: ("branch", "scope"), input_identity={})


def test_rejects_non_spd_hessian_and_nonfinite_direction():
    dtype = torch.float64
    c = torch.zeros(2, dtype=dtype)
    p = torch.ones(1, dtype=dtype)

    def objective(x, q):
        return -0.5 * x.square().sum()

    def score(x, q):
        return x[0] * q[0]

    with pytest.raises(RuntimeError, match="positive definite"):
        compute_local_response(
            objective, score, c, p, {"d": torch.ones_like(p)},
            branch_check=lambda x, q: ("branch", "scope"),
            input_identity={"case": "concave"},
        )
    with pytest.raises(ValueError, match="finite"):
        compute_local_response(
            lambda x, q: 0.5 * x.square().sum(), lambda x, q: x[0] * q[0],
            c, p, {"d": torch.tensor([float("nan")], dtype=dtype)},
            branch_check=lambda x, q: ("branch", "scope"),
            input_identity={"case": "nonfinite"},
        )


def test_zero_rhs_accepts_only_zero_true_transpose_residual():
    dtype = torch.float64
    c = torch.zeros(2, dtype=dtype)
    p = torch.ones(1, dtype=dtype)
    result = compute_local_response(
        lambda x, q: 0.5 * x.square().sum(),
        lambda x, q: 0.5 * q.square().sum(),
        c, p, {"d": torch.ones_like(p)},
        branch_check=lambda x, q: ("zero", "fixed"), input_identity={"case": "zero-rhs"},
    )
    assert result.true_adjoint_residual == 0.0
    assert result.pcg_iterations == 0


def test_nonfinite_score_is_rejected_before_gradient():
    _, _, _, control, p, objective, _, directions = _quadratic_case()
    with pytest.raises(ValueError, match="score"):
        compute_local_response(
            objective, lambda c, q: c.new_tensor(float("inf")), control, p, directions,
            branch_check=lambda c, q: ("branch", "scope"), input_identity={"case": "bad-score"},
        )


def test_true_transpose_residual_is_checked_even_when_pcg_reports_success(monkeypatch):
    from advar import local_response as module
    from advar.matrix_free import PCGResult
    _, _, _, control, p, objective, score, directions = _quadratic_case()
    bad = torch.ones_like(control)
    monkeypatch.setattr(
        module, "pcg",
        lambda *args, **kwargs: PCGResult(solution=bad, converged=True, iterations=1, relative_residual=0.0),
    )
    with pytest.raises(RuntimeError, match="true adjoint residual"):
        module.compute_local_response(
            objective, score, control, p, directions,
            branch_check=lambda c, q: ("branch", "scope"), input_identity={"case": "bad-adjoint"},
        )


def test_rejects_empty_vectors():
    empty = torch.empty(0, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonempty"):
        compute_local_response(
            lambda c, p: c.sum(), lambda c, p: p.sum(), empty, torch.ones(1, dtype=torch.float64),
            {"d": torch.ones(1, dtype=torch.float64)},
            branch_check=lambda c, p: ("branch", "scope"), input_identity={"case": "empty"},
        )
