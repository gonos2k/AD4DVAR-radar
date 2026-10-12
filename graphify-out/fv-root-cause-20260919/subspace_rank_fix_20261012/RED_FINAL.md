# RED final — rank-deficient two-direction residual model

The reported lambda-zero defect was reproduced before the main solver patch.
For `G=(-1,-1)`, `D=I`, `V=ones`, and `Delta=0.8`, the old implementation
entered a multiplier bisection despite an attainable least-squares residual zero
at `a=(0.5,0.5)`, then attempted to solve a singular matrix as the midpoint
rounded to lambda zero. `V=0` followed the same invalid path. No FV/HVP/score or
production routine was involved.

The current patch uses an SVD rank decision on normalized V, a Q-metric
rank-one solution-set representative, zero coefficients for rank zero, and QR
for the rank-two lambda-zero solve. The Q-metric representative is necessary:
in the constant-Hessian example in `RED_DESIGN.md`, the feasible zero-residual
step has movement `0.32732684`, while the Euclidean pseudoinverse representative
has movement `0.34641016`, outside the fixed `0.34` radius. The regression also
checks the case where the Q-minimum lies outside the positive orthant and the
axis solution is selected.

Independent new regression coverage passes: **11 passed in 1.58s**. It includes
the exact old failure, 26-control/27-residual padding, smooth quadratic gradient
and Hessian directions, rank-one active radius, zero V with finite inactive
branch gradients, rank-one fixed-factor gradchecks for both inactive and active
radii, and finite gradients on both sides of the near-rank threshold. This is
only the new regression file; the shared 11-test and 44-test baselines were not
repeated here.

The rank boundary is a finite-precision classification using
`64 * eps * max(shape) * largest_singular_value`. Discarding a singular value
below that threshold means the coefficient solve uses the effective rank-one
model. It does not make the underlying V exactly rank one, nor does it support a
gradient claim across the rank decision. The fixed-rank factor gradchecks cover
tensor derivatives within the selected branch; they do not cover rank changes.
The final candidate still needs actual J/R and repeat-P2/full-G gates, as in the
existing runner.

RED changed no solver/source file and ran no production model, HVP, optimizer,
or score. Main owns integration, full affected-test evidence, and the final
incremental Graphify refresh for the solver and this new test file.
