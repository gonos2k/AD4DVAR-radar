# PR301 rank and radius audit at cached d19e

## Result

The d19e two-column residual derivative matrix is **full rank**, so the new rank-one `lambda=0` path does not activate on this receipt. The current gradient-axis candidate is KKT-valid at its frozen radius. The second direction reaches its inactive-bound multiplier zero at a critical radius of `2.538864637100271e-5`, or `1.35354297` times the original radius `1.8757177916958825e-5`. A tightly bounded fresh-point expansion check near `1.5x` is mathematically worth considering as a diagnostic; its modeled gain is only `3.4053e-5` of the gradient-axis R. This local calculation does not justify expanding the production trust radius at a later point without fresh qualification.

## Archived inputs and method

`axis_radius_audit.py` validates the stored PR301 plan and source archive, then validates all 11 receipt members from `receipts.tar.gz` against `RECEIPT_MANIFEST.json`. It reads `fresh_models` and the d19e base `G` from the bundled `attempt1/step.json`; it does not use live endpoint loaders, FV evaluations, HVPs, or scores. The archived plan has 216 source pins and 287 archive pins; the raw child hash is `0802769b3dd6624c11e67ff9a222594f2ec25c728ea704021903419e64c6a4cb`.

For direction columns `D` and their corresponding residual derivatives `V`, the solver's normalization uses `s_j = ||D_j||`, `D_n = D diag(s)^{-1}`, `V_n = V diag(s)^{-1}`, and normalized coefficients `beta = diag(s) a`. The audited system is `M = V_n^T V_n`, `Q = D_n^T D_n`, and `c = V_n^T G`. At each radius the current local solver is swept offline; candidate output does not call the model.

## Rank and KKT values

The normalized residual matrix has singular values `[113.3936229262, 1.5855801015]`. The declared FP64 rank tolerance is `4.3508348128e-11`, so `rank(V_n)=2` and its singular-value condition ratio is `71.5155`. The Gram matrix condition number is `5114.473`; it is finite but makes the original normal-equation rank test numerically sensitive. The main solver fix checks rank in `V_n` and uses QR for the full-rank `lambda=0` solve.

At `Delta0 = 1.8757177916958825e-5`, the normalized axis solution is `beta=[1.8757177917e-5, 0]`. The radius constraint is active. Its KKT values are:

- Radius multiplier `lambda = 9417.5047694`.
- Inactive second-axis multiplier `mu1 = +0.06984030175`.
- Free-coordinate stationarity residual `0` to stored precision.
- Radius-equation residual `1.03e-25`.

The axis remains KKT-valid while both multipliers stay nonnegative within scale-aware sign budgets. Along the radius-active gradient axis, `mu1(beta0)` is linear. It reaches zero at `Delta_crit = 2.538864637100271e-5`; there `lambda=3599.1542771`, so the second axis activates before the gradient-axis multiplier reaches zero at `3.249526221387056e-5`. The gradient-axis active interval is therefore `[0, Delta_crit)` for positive radii. A sensitivity-scaled FP64 bracket is `[2.5388646363621617e-5, 2.5388646378383802e-5]`; the inactive multiplier changes sign from `+7.7735e-11` to `-7.7735e-11` across this bracket.

The full unconstrained positive `lambda=0` solution is `beta=[1.8744320918e-5, 0.02977026992]`, with movement radius `0.02978563336`, about 1,588 times the frozen radius. That unconstrained radius is not a reasonable expansion target.

## Radius sweep and bounded recommendation

The current solver returns the gradient axis through `1.35354x Delta0`. Just above the transition (`1.36708x`), it gives a small positive second coefficient `2.7250e-7`. At `1.5x Delta0`, the normalized coefficients are `[2.56744e-5, 2.94176e-6]`; the predicted residual-merit gain over the gradient axis is `8.6717e-8`, or `3.4053e-5` of the gradient-axis predicted R. At `2x`, the modeled gain is `1.4231e-6`, or `5.5890e-4` of the gradient-axis predicted R, but larger radius also increases exposure to chart curvature and branch changes. No finite candidate above `Delta0` is present in the receipts.

At the original radius, the archived actual candidate has `eta_vec=4.55746e-5`, `rho_R=1.0000313`, and its branch signature matches the base. This is strong local agreement at one small step. It supports at most one bounded **new-point diagnostic** with radius capped near `1.5x Delta0`, after requalifying the new base and rebuilding both directions. That test should retain chart-radius safeguards, original J/R approval, fresh paired HVPs, repeated P2/full-G, and no-expansion-after-accept policy. The d19e vectors alone do not support a production decision or claim that the subspace correction improves convergence.

## Rank-one `lambda=0` fix check

The separate rank-one fixture has identical residual columns and a feasible affine minimizer set. The minimum-control-norm nonnegative representative is `beta=[0.5, 0.5]`, with movement `0.70710678` under radius `0.8` and predicted R `1.97e-31`. The current solver reports `rank_one_interior`; regression tests also check a finite gradient through this branch and the radius-active rank-one solution. The implementation now applies the D metric through QR solves, avoiding formation of the ill-conditioned Gram solve for the rank-one representative. This validates the new lambda-zero branch separately from the full-rank d19e radius analysis.

The audit JSON records the receipt hashes, matrices, multipliers and scale-aware budgets, radius bracket, sweep, current solver hash `4a8b34378f139d39297a7e035666a00cd1a1cd91fc1cfccd771756f8d01c7242`, and rank-one check. `tests/test_fv_point_3h_subspace_radius_audit.py` passes five focused tests in 2.28 seconds; expansion-config Pyright reports zero errors and warnings for the audit script and test. No numerical source or raw receipt was edited by this audit task.
