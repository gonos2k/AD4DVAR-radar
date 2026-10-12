# RED diagnosis — rank-deficient residual model at an inactive radius

## Reproduction

Using the original solver snapshot and `.venv/bin/python`,

```python
G = tensor([-1, -1], float64)
D = eye(2, float64)
V = ones((2, 2), float64)
Delta = 0.8
```

gives a rank-one residual map. The residual is zero for every nonnegative
coefficient vector with `a0+a1=1`. The minimum control norm is attained at
`a=(0.5,0.5)`, whose norm is `sqrt(0.5)=0.70710678 < 0.8`. This is an inactive
radius, lambda-zero solution. The old solver skipped lambda-zero feasibility when
`VᵀV` was singular, entered multiplier bisection, and eventually evaluated the
singular lambda-zero endpoint. Reproduction raised
`torch._C._LinAlgError: ... input matrix is singular` at the midpoint solve in
`fv_point_3h_subspace_model.py` (old `solve_two_direction_subspace`, around line
148). `V=0` has the same defect: zero coefficients already minimize the model,
but the old path attempted the same singular solve.

The regression test file was run while the main agent was editing the shared
solver. The first old-snapshot reproduction failed at the singular solve; later
cases in that same pytest process may have loaded the new rank helper. Treat that
mixed run only as the original reproducer, not as post-fix verification. Main owns
the solver edit and final test run.

## Rank-one solution set and control metric

With normalized columns, write `V = u wᵀ`, `||u||=1`, and
`tau = -uᵀG`. Least-squares minimizers satisfy `wᵀ beta = tau`; they form an
affine line, rather than a unique vector. The control metric is
`Q = DᵀD`, so the minimum-control-norm representative is

`beta_Q = Q⁻¹ w * tau / (wᵀ Q⁻¹ w)`.

If `beta_Q` is componentwise nonnegative and `||D beta_Q|| <= Delta`, it is a
feasible unconstrained residual minimizer, and lambda must remain zero. If it is
positive but outside the ball, scaling it to the radius gives the closest
attainable point on that rank-one residual line. If a component is negative, the
minimum-metric point on the affine line lies outside the orthant; in two
coefficient dimensions, the constrained optimum must be represented by the zero
or an axis candidate. The axis cases must still be compared by the actual model
merit.

Do not substitute the Euclidean Moore-Penrose solution for `beta_Q`. A smooth
linear physical map demonstrates why. Let D's unit columns be
`(1,0)` and `(-1/2,sqrt(3)/2)`, let H be the constant one-row Jacobian
`(1,5/sqrt(3))`, and pad H/D to 26 controls and 27 residuals. Then `V=HD=(1,2)`
on its only active residual row, `G=(-1,0,...)`, and `Q=[[1,-1/2],[-1/2,1]]`.
The exact-fit minimum-control-norm coefficients are `(2/7,5/14)`, with movement
`sqrt(3/28)=0.32732684`. The Euclidean pseudoinverse gives `(0.2,0.4)`, whose
movement is `sqrt(0.12)=0.34641016`. At `Delta=0.34`, the correct metric solution
is feasible while the Euclidean representative is outside the radius.

## Minimal rank-aware path and differentiability

- Reveal the rank of `V`, not `VᵀV`; the latter squares the condition number.
  A tolerance scaled by dtype epsilon, matrix dimensions, and the largest
  singular value makes the rank branch explicit. This rank decision is a fixed
  active-set choice; do not differentiate through it.
- Rank two can retain lambda-zero least squares and the radius-active KKT path;
  use QR/back-substitution for lambda zero so near-rank-full matrices do not
  square their condition number.
- Rank one can factor through a largest nonzero column and use `Q` to select the
  least-control-norm affine representative. Compare the exact/interior or
  radius-active candidate against zero and both axes. If the metric representative
  lies outside the positive orthant, check the axis candidates explicitly.
- Rank zero (`V=0`) has no residual descent; return zero coefficients without
  forming `M + lambda Q` or evaluating inactive divisions by zero.
- Keep the differentiable tensor formulas for the selected fixed-rank branch.
  Gradients are piecewise valid away from rank and active-set transitions. Test
  finite derivatives of zero/axis/inactive branches and gradcheck a smooth
  rank-one manifold with respect to G and the metric inputs. Do not claim a
  derivative across a rank transition.

## Counterexamples and regression plan

`tests/test_fv_point_3h_subspace_rank_regression.py` covers:

1. The exact `G=(-1,-1)`, `D=I`, `V=ones`, `Delta=0.8` inactive rank-one case:
   coefficients `(0.5,0.5)`, zero predicted R, and movement below radius.
2. The smooth padded `V=HD` physical-Jacobian case above, distinguishing the
   Q-minimum from the Euclidean pseudoinverse.
3. The exact case padded to live `(26,2)` control directions and `(27,2)` residual
   directions.
4. Rank-one active radius `Delta=0.6`: coefficients
   `(Delta/sqrt(2),Delta/sqrt(2))`, boundary movement, and analytic predicted R.
5. A rank-one metric-minimum outside the positive orthant, where the radius-active
   optimum is the positive axis endpoint.
6. `V=0`, where zero is optimal and gradients of the selected inactive branch
   remain finite.
7. Samples on both sides of a near-rank-full threshold, requiring finite model
   outputs, feasible control movement, and finite G gradients.

The current shared solver is changing during this review. These are synthetic
linear-model checks only; no FV, HVP, score, forecast, or production comparison
was run. Existing baseline evidence remains separate and should not be repeated
as part of this rank fix.
