# RED design review: full coupled robust-GN tangent direction — 2026-10-10

## Scope and saved base

Design review only: no source edit, seed, FV, production gradient/HVP, test, guard, or Graphify run. The comparison base is the accepted three-step resume endpoint `c8fba1f93ee0b6c83edc5b56792de158921c0a64bab41eef5403c2a514b1fc33`, with native `J=0.06123349298793274`, fresh interior `theta*=0.4818866600367756`, `F²=0.004605971728402105`, `||g*||=0.06786730971831803`, `Q=0`, and side-gradient jump norm `0.28491830770495113`. The baseline and coupled-GN arms must start from this exact same saved state, fixed inputs, strict branch pair, face and theta policy.

## Coupled direction derivation

Let `nu=n/||n||`, `P=I−nu nuᵀ`, `gθ=(1−theta*)g_-+theta*g_+`, and `t=P gθ`. At a resolved interior theta-star, `j=g_+−g_-` satisfies `j·gθ=0`. Under the existing common-face support condition `j || n`, this gives `n·gθ=0` (up to the validated support/stationarity tolerance), hence the current normal multiplier estimate `mu*=n·gθ/(n·n)` is zero. This is an inner mixture stationarity identity at this point, not a constrained stationary-point certificate: `t` is still nonzero. The previous Jacobi construction's `−mu P Q_cc P` term should not be retained as a curvature correction here.

For the twelve fixed standardized/whitened pseudo-Huber observation residuals `r`, use the exact outer curvature

`D_i = rho''(r_i) = (delta / hypot(delta, r_i))^3 >= 0`

with the existing frozen masks and same selected-face extension. Let `A` be their `12×26` residual Jacobian and require projected row parity `A_- P ≈ A_+ P` before using one shared `A P`; equality of objective values alone does not establish row parity. With the original identity control-prior curvature, the tangent search metric is

`M_T = I_T + (A P)ᵀ D (A P)`.

Set `B = sqrt(D) A P`, `S = I_12 + B Bᵀ`. Then Woodbury gives

`(I + BᵀB)^−1 = I − Bᵀ (I + B Bᵀ)^−1 B`,

so the proposed direction is

`d = −t + Bᵀ S^−1 B t`.

This is the solution to the coupled tangent system without forming a `26×26` matrix or a tangent basis. `S` is symmetric positive definite with eigenvalues at least one; solve it with one `12×12` Cholesky/linear solve, never an explicit inverse. Since `B=B P`, both terms in `d` are tangent. For any nonzero tangent residual, `tᵀd = −tᵀ(I+BᵀB)^−1t < 0`. With `j || n`, both side objective slopes equal this negative tangent slope. This proves local native-J descent under the model assumptions; it does **not** prove descent of the squared-gradient merit.

This formula assumes the current frozen objective has identity control-prior curvature and no omitted nonzero linear-smoothing Hessian. If the actual frozen cost has another positive/linear curvature term, it belongs in the metric or the arm must be described as a robust-data GN search metric plus the identity prior; do not call it the full GN metric without confirming the cost decomposition at this base.

## Exact merit derivative remains separate

The GN metric defines only a search direction. For each arm, compute its own fresh two-sided exact objective HVP pair `h_-=H_-d`, `h_+=H_+d`; do not reuse the baseline direction's HVPs. Differentiate the actual interior argmin using

`theta' = −[j·h_mix + gθ·(h_+−h_-)]/(j·j)`,

where `h_mix=(1−theta*)h_-+theta*h_+`. Use the full residual direction

`DF = [h_mix + j theta', (n·d)/0.84]`.

At this `Q=0` tangent point, the scalar envelope slope simplifies to `F·DF=gθ·h_mix`, but `theta'` remains necessary for `||DF||²`, the bounded alpha model, and model-vs-actual diagnostics. Candidate theta-star must be freshly recomputed from each candidate's actual side gradients, strictly interior/resolved with no clipping. Keep actual native-J Armijo **and** actual fresh-minimum F²/Psi Armijo, current face/strict branch-pair gates, true chart-radius check, finite gradients, and per-side final repeat closure for both arms. Select at most one commit only after both arm searches are complete, using the frozen actual endpoint F²/Psi selection/tie rule; the metric or predicted decrease cannot add an acceptance gate.

A small counterexample shows why J descent cannot replace the F² gate. On one tangent coordinate let `J(y)=y−0.5y²` at `y=0`; then `g=1`, `H=−1`, and `Psi=0.5g²`. For any SPD scalar metric `M=1+b²`, `d=−M^−1g` satisfies `J'(0)d<0`, yet `Psi'(0)d=g H d=1/(1+b²)>0`. Thus a valid coupled-GN direction may still be refused by actual Psi/F² Armijo. A zero tangent residual or unresolved theta/jump also cleanly refuses this comparison; it is not a trigger to invent a new solver or boundary-minimum mode.

## Fixed comparison budget and limits

Use one same-point comparison: 24 row reverse products (12 per extension for row-parity verification), four fresh HVPs (two per arm), one dense `12×12` solve for the coupled arm, zero PCG and zero full/control-space Hessian. Search each direction independently with at most 16 actual candidates, one final commit maximum, internal 600 seconds, outer 660 seconds, and sampled RSS 1 GiB. Both directions use the same predeclared actual gates and tie policy; a rejected arm stays a refusal even if its prediction is attractive.

This is a one-point approximate direction comparison. Positive `M_T` proves the specified search metric is SPD, not that the exact constrained Hessian is positive, the squared-gradient merit decreases, the coupled direction is more effective, or the state is a root/minimum. The `rho'(r_i) Hessian(r_i)` residual-curvature terms remain omitted from the GN metric. No curvature, response, reanalysis, global smoothness, or forecast-skill claim follows.
