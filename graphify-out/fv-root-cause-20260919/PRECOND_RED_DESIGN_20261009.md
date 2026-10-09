# RED design review: one-point tangent Jacobi-preconditioner comparison — 2026-10-09

## Scope

This is a design review only. No source was changed, and no FV, production gradient, HVP, guard, test, or Graphify run was performed. The proposed comparison is at the saved final resume control `33cb86ca…`, `theta=0.3295008210283633`, `J=0.06124426330405451`, `F²=0.005145254919599806`, and `||F||=0.0717304323`. The remaining tangent-gradient norm is `0.0717279907`; both directions must start from this same state with the same fixed input, current branch pair, face, and native objective. The last HVP pair is based at `a579d422…`, so it is historical and cannot supply either new direction's HVP.

## Governing model and identities

For the fixed analysis problem, the native cost contains separable pseudo-Huber costs of the frozen whitened observation residuals, the original control prior, and linear field-smoothness residuals (`src/advar/variational.py:5060-5080, 5476-5505, 5905-5930, 6527-6550`). For one scalar standardized/whitened residual `r`,

`rho(r) = delta^2 (sqrt(1 + (r/delta)^2) - 1)`,
`rho''(r) = (delta / hypot(delta,r))^3`.

Thus a Gauss-Newton positive term from residual Jacobian row `a_i = grad r_i` is `D_i a_i a_i^T`, with `D_i=rho''(r_i)>=0`. This is the derivative of the robust score, not the existing frozen IRLS weight squared: `freeze_irls_weights` produces a squared weight `delta/hypot(delta,r)` (power one), so that buffer must not be reused as `D_i`. The scalar rows must be differentiated through the same frozen whitening transform and selected-face extension as the objective. Missing/invalid rows remain masked to zero; they are not clear-sky residuals.

Let `nu=n/||n||`, `P=I-nu nu^T`, and `Q(c)=sum_i w_i tanh(c_i)` be the unscaled active-face constraint. If the original control prior is exactly `0.5||c||²` and `L` is the linear field-smoothness Hessian, the projected positive Gauss-Newton/prior/smoothness diagonal is

`diag(P I P) + diag(P L P) + sum_i D_i (P a_i) elementwise-squared`.

The face-curvature matrix for `L(c)=J(c)-mu Q(c)`, with `mu=(n·g_mix)/(n·n)`, is `-mu P Q_cc P`. For `q_i=(Q_cc)_ii=-2 w_i tanh(c_i)(1-tanh²(c_i))`, its projected diagonal is exactly

`diag(P Q_cc P)_i = q_i (1-2 nu_i²) + nu_i² sum_j nu_j² q_j`.

I checked this identity with synthetic FP64 values; it matched direct `diag(P @ Q_cc @ P)` to `2.3e-16`. The proposed `B_i` can nevertheless be negative after subtracting the curvature term, even for the selected tanh face. For a synthetic face point with the actual weights `[0,-.08,-.28,-.14,-.84]` and `tanh(c[20:25])=(0,-.5,-.5,-.5,.25/.84)`, `Q≈2.8e-17`. With only the original prior diagonal and synthetic `mu=-20`, the unclipped flow entries of `B` are approximately `[1,-.148,-2.150,-.909,-.133]`. This is not the current endpoint's multiplier; it demonstrates why an SPD floor is needed in the formula. Anchor the floor to a positive, per-coordinate scale such as the projected original-prior diagonal plus its own dtype-scaled curvature epsilon; do not use `finfo.tiny` alone. Check finite inputs and record which coordinates hit the floor. Clipping is a preconditioner safeguard only; it must not change the native J or be described as regularizing the analyzed objective.

With finite `W=diag(1/max(B_i,floor_i))` and every `W_i>0`, `d=-P W P g_mix` is a valid tangent descent direction whenever `P g_mix != 0`: `n·d=0` and `g_mix·d=-(P g_mix)^T W(P g_mix)<0`. If `g_+-g_-` is normal to the common face, both side slopes equal that mixture slope. This is the key SPD property. Keep the current common-face jump check and both actual side-slope checks; if row/gradient parity is poor, an SPD matrix alone does not establish that both nonsmooth sides descend.

For example, with `n=(1,0)`, `theta=.5`, `g_-=(0,3)`, and `g_+=(0,-1)`, the mixed tangent gradient is `(0,1)` and its identity-preconditioned direction is `(0,-1)`: mixture slope `-1`, but the plus-side slope is `+1`. The gradient jump has a tangent component and violates common-face support. This counterexample is excluded by the current face contract; it shows why the current support and both-side descent checks must remain in force.

## Counterexamples and limits

The diagonal model is an approximate direction preconditioner, not an exact Hessian or a guarantee of better one-step progress. A synthetic quadratic makes that limit explicit: for `H=diag(1,100)`, `c=(1,1)`, and `g=Hc`, baseline steepest descent with exact scalar line minimization lowers J from `50.5` to `0.49005`. The still-positive but poorly scaled SPD metric `W=diag(1,.001)` also descends but reaches only `J=20.25`. The proposed metric may be better aligned than this deliberately bad example, but no theorem guarantees it; compare accepted endpoint `F²` at the same base, after both variants independently pass the existing actual-J/F² and fresh-closure checks.

The robust Gauss-Newton term also omits residual-curvature contributions `rho'(r_i) Hessian(r_i)`. Synthetic `r(x)=x-x²` at `x=.5`, `delta=2` has `r'=0`, so the positive GN term is zero, while the exact robust scalar Hessian is about `-0.49614`. Thus the floor plus `B` cannot certify positivity of the exact face Hessian, even before minmod branch curvature and the current nonzero tangent residual are considered. In this fixed task, `mu` is only the normal projection of a nonstationary gradient (`||g_T||≈0.07173`), not a KKT multiplier established at a stationary point. The `-mu Q_cc` term is a heuristic local scale correction here; its diagonal and any eigenvalue of `B` must not be reported as constrained curvature or a minimum test.

There is no general scale invariance for `W=cI`. Scaling the control direction by `c` scales its HVP, but the normal correction is `delta_theta(c) = -[nu·g_mix + c nu·(H_mix d)]/(nu·(g_+−g_-))`; the fixed `nu·g_mix` term does not scale. The alpha cap at one and the radius/merit cap can also select different endpoints. The helper regression tests `W=I` reproducing the current baseline only; it does not assert that other scalar multiples produce equivalent candidates.

For matching side residual values and projected residual rows, the projected positive GN data term can agree across the two face extensions. That does not make the full preconditioner theta- or extension-invariant: `mu` depends on the mixture, and the approximate diagonal omits `sum_i rho'(r_i) Hessian(r_i)`, the terms that participate in exact extension cancellation with `mu Q_cc`. Theta-dependent differences in this defined SPD search metric are therefore possible and acceptable as part of a heuristic direction comparison. Do not call `B` an exact intrinsic/Riemannian Hessian or claim side- or theta-invariance of its diagonal.

## Design conditions before implementation

- Pin the same `33cb86ca…` control, theta, native J, fixed parameters/truth, `Q=0`, and current paired branch trace for both direction variants. Recompute current `g_-`, `g_+`, and `g_mix`; do not reuse an old direction or HVP.
- Compute the 12 current standardized/whitened measurement residual values and each side's 12 scalar-gradient rows `A_-`, `A_+` at that point. This is 24 reverse-gradient products (not HVPs). Confirm row masks and value parity, and report projected row/aggregate parity `P A_-` versus `P A_+` within scale-aware roundoff. Objective scalar parity alone does not prove each indexed residual row is matched. If this premise fails, do not silently substitute native tie-autodiff or average unexplained rows.
- Specify the side combination before coding. If using a convex side-matrix blend, form the side outer products first: `sum_i [(1-theta) D_i^- (P a_i^-)² + theta D_i^+ (P a_i^+)²]`. Do not average row vectors and then square unless projected row equality has been established. Under exact common-face residual parity the projected positive terms should agree; theta should not create a new cost/objective.
- Confirm the fixed point uses the original `0.5||control||²` prior (no neural-prior residual), and that the field-smoothness residual is linear and its diagonal is the relevant `diag(P L P)`. In this control layout the face normal lives in the five flow coordinates; verify that the smoothness block has no unsupported overlap before simplifying `diag(P L P)` to `diag(L)`.
- Recompute `q_i`, `mu`, and the floor at the same base in FP64. Use a stable `hypot` for `D_i`; apply fixed observation masks before forming rows. Check `B`, floor, W, direction, norm, and tangent residual for finite values. A zero tangent gradient is a refusal in this comparison; theta-only cleanup remains separate.
- For each direction—baseline `-P g_mix` and preconditioned `-P W P g_mix`—compute its own fresh minus/plus HVP pair (4 HVPs total), its own model theta correction, and its own bounded chart search. Do not reuse baseline HVPs for the preconditioned direction. Preserve the current radius, candidate budget, theta domain, dual actual Armijo, branch-pair, side/native objective, face, finite-gradient, and fresh per-side closure checks.
- Choose the comparison winner only from independently accepted, freshly closed endpoints by the predeclared actual endpoint `F²` rule. Model `B`, predicted `F²`, or positive diagonals do not add an acceptance gate. A refusal is reported as a refusal, not as a poor endpoint.
- The proposed cost is 24 scalar reverse gradients plus 4 current HVPs and up to two existing bounded candidate grids under the stated 600-second internal / 660-second outer / 1-GiB sampled limits. The prior resume's 79.278-second, 565-MB run did not perform these scalar-row gradients, so it does not establish the new cost or memory peak. Keep all 24 row gradients and 4 HVPs separately counted; the performance budget remains a risk to verify in the authorized run.

No implementation, solver replacement, new stopping/acceptance threshold, condition-number estimate, curvature certificate, theta-only cleanup, or production run is proposed by this RED design review. Any later curvature comparison and theta-only final audit remain separate tasks.
