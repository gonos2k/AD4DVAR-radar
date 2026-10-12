# RED design: stable active rank-two subspace solve

## Reproducer and independent oracle

The pinned `solve_two_direction_subspace` case uses

```text
H = [[1, -1], [-1, 1 + 1e-8]],  G = -u_min(H),  D = I,  radius = 0.8
```

`H` is positive definite with eigenvalues approximately `2` and `5e-9`.
The exact unconstrained coefficient vector is positive and has norm about
`2e8`, so this is a full-rank, radius-active case. For radius `r`, the KKT
solution is `beta = r*u_min`, the multiplier is
`lambda = mu/r - mu**2`, and predicted squared residual is `(1-r*mu)**2`,
where `mu` is the small eigenvalue. The regression computes `mu` and `u_min`
from the *represented binary64* bottom-right entry with 80-digit `Decimal`,
then checks the padded `(27 residual, 26 control, 2 direction)` layout.

In the observed pre-patch LocalTorch 2.13 build, this input did not throw. It
returned movement `0.7999999881891453` instead of `0.8`, and the first
coefficient missed the independent value by `8.35e-9`. An augmented-QR patch
improved the observed movement error to about `2.0e-12` before the active-root
and derivative refinements were complete. The local matrix `V.T @ V` has a
small eigenvalue around `2.5e-17`, below the rounding uncertainty from
subtracting entries of order one. The discrepancy is a numerical-accuracy
failure from squaring `cond(V)` in the multiplier solve; it is not evidence of
a source-pinned exception, rank-one case, or orthant-boundary case. Exact
failure mode and values can vary with BLAS and torch versions.

After the main solver's augmented-QR and active-radius closure changes, both
`1e-8` and `1e-9` perturbations return movement `0.8` to displayed precision;
the scaled original gradient `G=(-1e-8,-1e-8)` also returns movement
`0.7999999999999999`. Their coefficients agree with the represented-input
Decimal references at tolerance scaled by `eps * cond(A(lambda))`. The
fixed-eigenvector scale gradcheck passes, as do finite-gradient checks on the
near-singular lambda-zero branch.

## Stable coefficient and multiplier calculations

For each positive trial multiplier, form the augmented least-squares system

```text
A(lambda) = [ V       ]       b = [-G]
            [ sqrt(l) D ]           [ 0]
```

and solve it with reduced QR, `A = Q R`, then triangular solves
`R beta = Q.T b`. This represents the stationarity equation
`(V.T V + lambda D.T D) beta = -V.T G` without constructing either normal
matrix. The movement `||D beta||` is monotone decreasing in positive lambda;
bracket the first feasible multiplier and bisect its scalar radius residual.
The initial positive multiplier must be clamped away from zero, while the
upper bracket grows until movement is feasible. Never evaluate the singular
lambda-zero endpoint on this active path.

At a bracketed root, set `Qbeta = D.T @ (D @ beta)` and compute
`w = (V.T V + lambda D.T D)^-1 Qbeta` through the same triangular factor:
solve `R.T z = Qbeta`, then `R w = z`. The derivative of the squared-radius
constraint is `-2 * (D beta).T @ (D w)`. One or more safeguarded Newton
corrections of lambda restore the implicit derivative of the active KKT root;
the final beta must be recomputed from QR at corrected lambda. Reject a
nonfinite or nonpositive correction and retain a bracket-feasible endpoint.
This also avoids differentiating through branch tests or a fixed count of
bisection decisions.

The calculation of `Qbeta` is a matrix-vector application, not a factorization
or solve of `Q`. For inactive divisions, use a finite denominator in the
unselected branch before `where`; the rank-zero/zero-curvature case must keep
finite derivatives as well.

## Derivative and boundary evidence

The active fixture checks an invariant smooth path: scale `G` along its fixed
small-eigenvector direction. The exact ball-constrained beta is invariant to
that positive scale, so its directional derivative is zero. This catches a
wrong implicit-lambda derivative while avoiding the narrow orthant active-set
transition produced by perturbing an ill-conditioned eigenvector. Perturbing
all entries of `G` or `V` at `1e-6` can cross that transition; no gradient claim
should be made across it.

The companion inactive test uses `beta=(0.2, 0.1)` and `G=-H beta`, checks the
full-rank lambda-zero branch, and requires finite gradients. Its coefficient
tolerance (`2e-8`) scales with `eps * cond(H)`, unlike the active exact
eigenvector/radius checks, whose Decimal reference supports near-machine
precision tolerances. Existing complementary coverage is in
`tests/test_fv_point_3h_subspace_model.py` (full-rank identity/skew gradchecks)
and `tests/test_fv_point_3h_subspace_rank_regression.py` (fixed-rank rank-one
active/inactive gradchecks, padded inactive case, and zero-derivative branches).

These tests are local linear-model checks only. They make no claim about FV,
HVP, score, forecast accuracy, or production model validity.

Final focused verification: `.venv/bin/python -m pytest
tests/test_fv_point_3h_active_qr_regression.py
tests/test_fv_point_3h_subspace_model.py
tests/test_fv_point_3h_subspace_rank_regression.py -q` passed all 27 tests.
