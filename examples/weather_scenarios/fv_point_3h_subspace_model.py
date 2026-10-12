"""Small differentiable model for a nonnegative two-direction residual step."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class SubspaceModel:
    """Result of minimizing a two-column linear residual model."""

    coefficients: Tensor
    direction: Tensor
    residual_direction: Tensor
    predicted_R: Tensor
    diagnostics: dict[str, Tensor | bool | str]


def _finite_vector(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have shape {shape} and finite values")


def _candidate_R(G: Tensor, V: Tensor, beta: Tensor) -> Tensor:
    return torch.linalg.vector_norm(G + V @ beta).square()


def _rank_one_candidate(G: Tensor, D: Tensor, V: Tensor,
        radius: float) -> tuple[Tensor | None, str]:
    # For V=u w^T, every least-squares solution satisfies w^T beta=-u^T G.
    # Its minimum control-norm representative uses Q, not coefficient norm.
    pivot = torch.argmax(torch.linalg.vector_norm(V, dim=0))
    u = V[:, pivot] / torch.linalg.vector_norm(V[:, pivot])
    w = V.T @ u
    # QR applies the control metric without squaring D's condition number.
    _, triangular = torch.linalg.qr(D, mode="reduced")
    metric_w = torch.linalg.solve_triangular(triangular.T, w.reshape(2, 1), upper=False)
    inverse_w = torch.linalg.solve_triangular(triangular, metric_w, upper=True).reshape(2)
    curvature = metric_w.square().sum()
    if not bool(torch.isfinite(curvature)) or bool(curvature <= torch.finfo(G.dtype).tiny):
        raise ValueError("rank-one residual curvature is numerically unresolved")
    beta = -torch.dot(u, G) * inverse_w / curvature
    if not bool(torch.isfinite(beta).all()):
        raise ValueError("rank-one least-squares solution is nonfinite")
    if not bool((beta >= 0.0).all()):
        # In two dimensions the nearest nonnegative member of the affine
        # solution set is on an axis. The caller already enumerates both axes.
        return None, "rank_one_orthant_boundary_axis"
    movement = torch.linalg.vector_norm(D @ beta)
    if bool(movement <= radius):
        return beta, "rank_one_interior"
    return beta * (radius / movement), "rank_one_radius_active"


def solve_two_direction_subspace(
    G: Tensor,
    D: Tensor,
    V: Tensor,
    radius: float,
    *,
    dependence_rtol: float | None = None,
    bisection_steps: int = 64,
) -> SubspaceModel:
    """Solve ``min_{a>=0} 0.5||G + V a||²`` subject to ``||D a||<=radius``.

    ``D`` contains two control-space tangent directions and ``V`` contains the
    corresponding residual-space directional derivatives in the same raw
    scaling. Columns are normalized by their control norm, with the same scale
    applied to ``V``. The returned coefficients are mapped back to the raw
    columns. Near same-ray dependence reduces the solve to the two axis
    candidates; near-antiparallel dependence is refused because cancellation
    could produce a false residual improvement at negligible control motion.

    Tensor algebra, including an implicit Newton correction of the active-radius
    multiplier, remains differentiable with respect to tensor inputs away from
    active-set changes. Scalar branch decisions select the active set.
    """
    if G.ndim != 1 or G.numel() == 0 or D.ndim != 2 or V.ndim != 2:
        raise ValueError("G must be a nonempty vector and D/V must be matrices")
    if D.shape[1] != 2 or V.shape != (G.numel(), 2) or D.shape[0] == 0:
        raise ValueError("D and V must contain exactly two matching direction columns")
    if (D.dtype != G.dtype or V.dtype != G.dtype or D.device != G.device
            or V.device != G.device or not G.is_floating_point()):
        raise ValueError("G, D, and V must share a floating dtype and device")
    _finite_vector("G", G, (G.numel(),))
    _finite_vector("D", D, (D.shape[0], 2))
    _finite_vector("V", V, (G.numel(), 2))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if bisection_steps < 1:
        raise ValueError("bisection_steps must be positive")

    eps = torch.finfo(G.dtype).eps
    effective_dependence_rtol = (
        max(1e-8, 64.0 * eps * max(D.shape))
        if dependence_rtol is None else dependence_rtol)
    if not math.isfinite(effective_dependence_rtol) or effective_dependence_rtol <= 0.0:
        raise ValueError("dependence_rtol must be finite and positive")

    scales = torch.linalg.vector_norm(D, dim=0)
    if not bool(torch.isfinite(scales).all()) or bool((scales <= torch.finfo(G.dtype).tiny).any()):
        raise ValueError("both control directions must have finite nonzero norm")
    Dn = D / scales
    Vn = V / scales
    if not bool(torch.isfinite(Dn).all() and torch.isfinite(Vn).all()):
        raise ValueError("direction normalization produced nonfinite columns")
    Q = Dn.T @ Dn
    correlation = torch.dot(Dn[:, 0], Dn[:, 1]).clamp(min=-1.0, max=1.0)
    orthogonal_part = Dn[:, 1] - correlation * Dn[:, 0]
    angular_separation = torch.linalg.vector_norm(orthogonal_part)
    dependent = bool(angular_separation <= effective_dependence_rtol)
    if dependent and bool(correlation < 0.0):
        raise ValueError("near-antiparallel directions could cancel control motion")

    # Work in normalized column coordinates beta = scales * a. The feasible
    # axis interval is then [0, radius] for either unit control direction.
    candidates: list[tuple[str, Tensor]] = [("zero", G.new_zeros(2))]
    for index, name in enumerate(("axis_0", "axis_1")):
        vi = Vn[:, index]
        curvature = torch.dot(vi, vi)
        linear = torch.dot(G, vi)
        raw = torch.where(curvature > 0.0, -linear / curvature.clamp_min(torch.finfo(G.dtype).tiny), 0.0)
        beta_i = raw.clamp(min=0.0, max=radius)
        beta = G.new_zeros(2)
        beta = beta.scatter(0, G.new_tensor([index], dtype=torch.long), beta_i.reshape(1))
        candidates.append((name, beta))

    singular_values = torch.linalg.svdvals(Vn)
    rank_tolerance = 64.0 * eps * max(Vn.shape) * singular_values[0]
    residual_rank = int((singular_values > rank_tolerance).sum())
    full_status = "dependent_axis_fallback" if dependent else "full_2d_not_needed"
    full_beta: Tensor | None = None
    if residual_rank == 0:
        full_status = "zero_residual_derivative"
    elif not dependent and residual_rank == 1:
        full_beta, full_status = _rank_one_candidate(G, Dn, Vn, radius)
    elif not dependent:
        M = Vn.T @ Vn
        c = Vn.T @ G
        if not bool(torch.isfinite(M).all() and torch.isfinite(c).all()):
            raise ValueError("residual model normal equations overflowed")
        # If the unconstrained normal equations have a unique positive solution
        # inside the movement ball, it is the full KKT minimizer (lambda=0).
        m_scale = torch.clamp_min(torch.linalg.vector_norm(M), torch.finfo(G.dtype).tiny)
        # Rank was revealed in V itself. QR avoids squaring its condition number
        # when checking the unconstrained lambda=0 solution.
        orthogonal, triangular = torch.linalg.qr(Vn, mode="reduced")
        beta0 = torch.linalg.solve_triangular(triangular,
            -(orthogonal.T @ G).reshape(2, 1), upper=True).reshape(2)
        radius0 = torch.linalg.vector_norm(Dn @ beta0)
        if bool((beta0 >= 0.0).all()) and bool(radius0 <= radius):
            full_beta = beta0
            full_status = "full_2d_interior"
        needs_radius_solve = bool(radius0 > radius)
        if full_beta is None and needs_radius_solve:
            # For an active movement constraint, stationarity is
            # (V'V + lambda D'D) beta = -V'G. Bracket lambda, then check that
            # the radius-active solution lies inside the positive orthant.
            lam = torch.maximum(m_scale * eps, G.new_tensor(torch.finfo(G.dtype).tiny))
            lam_lo = G.new_zeros(())
            lam_hi = lam
            bracketed = False
            for _ in range(80):
                beta_hi = torch.linalg.solve(M + lam_hi * Q, -c)
                r_hi = torch.linalg.vector_norm(Dn @ beta_hi)
                if bool(torch.isfinite(r_hi)) and bool(r_hi <= radius):
                    bracketed = True
                    break
                lam_lo = lam_hi
                lam_hi = lam_hi * 2.0
                if not bool(torch.isfinite(lam_hi)):
                    break
            if bracketed:
                for _ in range(bisection_steps):
                    lam_mid = 0.5 * (lam_lo + lam_hi)
                    trial = torch.linalg.solve(M + lam_mid * Q, -c)
                    trial_radius = torch.linalg.vector_norm(Dn @ trial)
                    if bool(trial_radius > radius):
                        lam_lo = lam_mid
                    else:
                        lam_hi = lam_mid
                # Bisection chooses the active branch. Newton corrections then
                # restore the implicit derivative d(lambda)/d(M,c) from the
                # active norm equation instead of differentiating branch tests.
                lam_star = lam_hi
                beta_mid = torch.linalg.solve(M + lam_star * Q, -c)
                for _ in range(2):
                    system = M + lam_star * Q
                    beta_mid = torch.linalg.solve(system, -c)
                    inverse_Q_beta = torch.linalg.solve(system, Q @ beta_mid)
                    norm_residual = torch.dot(beta_mid, Q @ beta_mid) - radius * radius
                    norm_derivative = -2.0 * torch.dot(beta_mid, Q @ inverse_Q_beta)
                    if (not bool(torch.isfinite(norm_derivative))
                            or bool(norm_derivative.abs() <= torch.finfo(G.dtype).tiny)):
                        break
                    corrected = lam_star - norm_residual / norm_derivative
                    if not bool(torch.isfinite(corrected)) or bool(corrected <= 0.0):
                        break
                    lam_star = corrected
                beta_mid = torch.linalg.solve(M + lam_star * Q, -c)
                if bracketed and bool((beta_mid >= 0.0).all()):
                    beta_radius = torch.linalg.vector_norm(Dn @ beta_mid)
                    radius_tolerance = 128.0 * eps * max(radius, torch.finfo(G.dtype).tiny)
                    if bool(beta_radius <= radius + radius_tolerance):
                        full_beta = beta_mid
                        full_status = "full_2d_radius_active"
                    else:
                        # Keep the feasible bracket endpoint if a last-bit
                        # Newton correction rounded just outside the ball.
                        full_beta = torch.linalg.solve(M + lam_hi * Q, -c)
                        full_status = "full_2d_radius_active_bracket_endpoint"
                elif bracketed:
                    full_status = "full_2d_orthant_boundary_axis"
            else:
                raise ValueError("failed to bracket a feasible radius-active KKT multiplier")

    if full_beta is not None:
        candidates.append(("full_2d", full_beta))

    values = torch.stack([_candidate_R(G, Vn, beta) for _, beta in candidates])
    selected_index = torch.argmin(values)
    # The index is a discrete active-set decision; indexing keeps the selected
    # candidate's differentiable tensor graph intact.
    beta_stack = torch.stack([beta for _, beta in candidates])
    beta_selected = beta_stack[selected_index]
    selected_slot = int(selected_index)
    selected_name = candidates[selected_slot][0]

    coefficients = beta_selected / scales
    direction = D @ coefficients
    residual_direction = V @ coefficients
    predicted_R = torch.linalg.vector_norm(G + residual_direction).square()
    movement_norm = torch.linalg.vector_norm(direction)
    diagnostics: dict[str, Tensor | bool | str] = {
        "selected_active_set": selected_name,
        "full_kkt_status": full_status,
        "reduced_domain": dependent,
        "control_columns_near_dependent": dependent,
        "residual_derivative_rank": G.new_tensor(residual_rank),
        "residual_rank_tolerance": rank_tolerance,
        "movement_norm": movement_norm,
        "radius": G.new_tensor(radius),
        "predicted_R_before": torch.dot(G, G),
        "predicted_R_after": predicted_R,
    }
    return SubspaceModel(coefficients, direction, residual_direction, predicted_R, diagnostics)


def residual_model_fidelity(G: Tensor, predicted_residual: Tensor,
                            actual_residual: Tensor) -> dict[str, Tensor]:
    """Compare a predicted residual vector with the actual vector at a trial.

    Reports full-vector error, change-vector error, and squared-merit error.
    The denominators are the actual endpoint norms (or dtype tiny at zero), so
    the two vector errors remain distinct when the step change is small.
    """
    if (G.ndim != 1 or predicted_residual.shape != G.shape
            or actual_residual.shape != G.shape or G.dtype != predicted_residual.dtype
            or G.dtype != actual_residual.dtype or G.device != predicted_residual.device
            or G.device != actual_residual.device):
        raise ValueError("G and predicted/actual residuals must be matching vectors")
    _finite_vector("G", G, G.shape)
    _finite_vector("predicted_residual", predicted_residual, G.shape)
    _finite_vector("actual_residual", actual_residual, G.shape)
    tiny = torch.finfo(G.dtype).tiny
    full_error = torch.linalg.vector_norm(predicted_residual - actual_residual)
    actual_norm = torch.linalg.vector_norm(actual_residual)
    predicted_delta = predicted_residual - G
    actual_delta = actual_residual - G
    delta_error = torch.linalg.vector_norm(predicted_delta - actual_delta)
    actual_delta_norm = torch.linalg.vector_norm(actual_delta)
    predicted_R = torch.dot(predicted_residual, predicted_residual)
    actual_R = torch.dot(actual_residual, actual_residual)
    return {
        "full_vector_error": full_error,
        "full_vector_relative_error": full_error / actual_norm.clamp_min(tiny),
        "delta_vector_error": delta_error,
        "delta_vector_relative_error": delta_error / actual_delta_norm.clamp_min(tiny),
        "predicted_R": predicted_R,
        "actual_R": actual_R,
        "absolute_R_error": (predicted_R - actual_R).abs(),
        "relative_R_error": (predicted_R - actual_R).abs() / actual_R.abs().clamp_min(tiny),
    }
