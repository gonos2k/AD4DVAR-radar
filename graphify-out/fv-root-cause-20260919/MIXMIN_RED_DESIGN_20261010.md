# RED design review: interior minimum-norm side-gradient mixture — 2026-10-10

## Scope and context

This is a math/design note only. No source was changed, and no seed, FV, production gradient/HVP, guard, or test was run. The proposed priority-1 mode changes the merit from the carried-theta residual to the scalar envelope

`Psi(c) = min_{0 < theta < 1} 0.5 * ||[(1-theta) g_-(c) + theta g_+(c), Q(c)/0.84]||²`

at the already accepted baseline endpoint `0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1`. Its saved state is `theta=0.3298981720696071`, `J=0.06124393564803218`, and carried-theta `F²=0.005132555144362135`. The previous comparison archive contains an unapplied side-gradient-only `theta_min=0.3277376621747043`, `F²=0.005132174540958916` at that control. That saved arithmetic is historical context only: a new mode must freshly reconstruct the base and recompute its interior theta-star under the frozen live source. A fresh base normalization is not a control/optimizer step and must not replace the carried theta if the run refuses.

Keep the original native J, prior, face `Q=0`, fixed inputs, current paired branch, and baseline tangent direction. This is a new scalar merit definition; it does not make earlier PR270/F² line-search refusals accepted retroactively. It is also not a preconditioned direction comparison, curvature test, minimum claim, response, or forecast result.

## Interior theta-star and envelope derivative

Let `j=g_+−g_-`. At fixed `c`, the face term is independent of theta, and the unconstrained scalar minimizer is

`theta_star = -(g_-·j)/(j·j)`,
`g_star = g_- + theta_star*j`.

This value is admissible only when `j` is resolved, the common-face jump check passes, and the minimizer is strictly interior with a dtype/gradient-scale boundary margin. Do not clamp a value outside `[0,1]` and call it an interior solution. Refuse exact or numerically unresolved endpoints at `0` or `1`; handle boundary active-set solutions only under a later explicit mode. If `j·j` is unresolved/zero, theta is nonunique or unstable and the mode must refuse rather than divide by it. The candidate endpoint must repeat the same checks using fresh side gradients.

Under the current face contract, `j` is normal to `Q=0`. Therefore interior theta-star removes the mixture's normal component, while `P g_star` is the remaining tangent residual. Minimizing over theta cannot remove tangential stationarity error; the current saved tangent norm at the baseline endpoint is still about `0.07164`. This normalization alone does not qualify a minimum.

For a tangent control direction `d`, define current side HVPs `h_- = H_- d`, `h_+ = H_+ d`, `h_mix=(1-theta_star)h_-+theta_star h_+`. Differentiating the interior stationarity condition `g_star·j=0` gives

`theta_prime = -[j·h_mix + g_star·(h_+−h_-)]/(j·j)`.

Thus the full residual direction is

`DF = [h_mix + j*theta_prime, (n·d)/0.84]`.

The envelope scalar slope is

`F·DF = g_star·h_mix + (Q/0.84²)*(n·d)`,

because the `theta_prime*(g_star·j)` term vanishes at an interior theta-star. At `Q=0` and a tangent `d`, this is `g_star·h_mix`. The theta-prime term still matters to `||DF||²` and the quadratic residual-norm alpha model. Omitting it can leave the first scalar slope correct while producing the wrong alpha cap and model prediction.

## Synthetic derivative counterexample

Consider the synthetic face `Q(x,y)=x=0` and side extensions

`J_-(x,y)=h(y)+(-0.2+0.4y)x`,
`J_+(x,y)=h(y)+( 0.2+0.6y)x`,
`h(y)=y+0.5y²`.

Both sides restrict to the same objective on `x=0`. At `(0,0)`, `g_-=(-0.2,1)`, `g_+=(0.2,1)`, `j=(0.4,0)`, `theta_star=0.5`, `g_star=(0,1)`, and the baseline tangent direction is `d=(0,-1)`. The side HVP mixture is `h_mix=(-0.5,-1)` and `h_+−h_-=(-0.2,0)`, so `theta_prime=1.25` and full `DF=(0,-1)`. A synthetic `torch.func.jvp` of the complete minimized residual agrees with this vector exactly; the envelope slope is `-1`.

If theta-prime is omitted, `DF_wrong=(-0.5,-1)`: its scalar `F·DF` remains `-1`, but its squared norm is `1.25` instead of `1`, moving the model-optimal alpha from `1` to `0.8`. At alpha `0.2`, the actual point remains interior (`theta_star=0.77778`) and `Psi` falls from `0.5` to `0.32`; the full model predicts the actual `F²=0.64`, while the incomplete direction predicts `0.65`. This is a small analytic counterexample, not an FV run.

Boundary and unresolved cases are separate refusals. With the same `Q=x` and normal-only jumps, `g_-=(1,1), g_+=(2,1)` gives unconstrained `theta_star=-1`; `g_-=(-2,1), g_+=(-1,1)` gives `theta_star=2`. Exact boundary examples are `g_-=(0,1), g_+=(1,1)` for `theta_star=0` and `g_-=(-1,1), g_+=(0,1)` for `theta_star=1`; all four jumps are normal to the face. These cases must refuse in this interior-only mode, rather than being clipped. If `g_+=g_-`, then `j·j=0` and theta-star is not identified. These cases need no HVP and must leave the stored base control/theta unchanged.

## Candidate path and transaction contract

At the freshly closed baseline base, recompute current native J, both side gradients, traces, current branch pair, face value, input/runtime/source, and carried-theta F² against the prior receipt before using theta-star. Record `theta_star` and `Psi(base)` as base diagnostics, while retaining the original control/theta as the last-confirmed state until a real nonzero control candidate is accepted.

For an eligible interior base, use only the existing baseline tangent control direction `d=-P g_star`. Compute exactly two fresh current side HVPs along this `d`; do not reuse the previous endpoint's HVPs or the diagonal preconditioner direction. Use the full `DF` above for the bounded quadratic norm-model alpha cap, with the existing radius/candidate policy. At each actual candidate, independently observe J, both side gradients, face/branch pair, and calculate a fresh interior `theta_star(candidate)` from those gradients. The actual merit is `Psi(candidate)` at that newly minimized theta, not the base theta plus a linear theta step. Keep native-J Armijo, actual `Psi`/F² Armijo, current strict pair/face gates, and the existing fresh per-side final closure. Only after the chosen nonzero-control proposal passes the independent repeat may the control and theta-star commit together. If there is no accepted candidate or any boundary/unresolved/closure refusal, keep the original carried control and theta; do not publish base theta normalization as an optimizer step.

The candidate path remains on `Q=0` through the current implicit chart and actual radius check. Branch signatures may change between base and candidate; only current two-side paired endpoint qualifications are evidence. The new mode does not establish global smoothness, a nonsmooth minimum, response, reanalysis, or forecast skill.

## Fixed execution bounds and open work

The proposed one-root-guard policy is 240 seconds internal, 300 seconds outer, 1-GiB sampled RSS, two current HVPs, zero row-Jacobian VJPs, zero PCG/dense solves, and the existing 16-candidate line-search cap. An early base/candidate theta boundary or unresolved jump is a clean refusal; there is no forced clipping or silent switch to an endpoint mixture.

No code, plan, result, or launch was created by this review. A separate GREEN/code review and fixed plan should verify the complete `torch.func` derivative of `Psi`, the theta-prime alpha-minimization counterexample, theta boundary/unresolved refusal tests, the fresh base identity, and that a refused candidate preserves the old carried theta. The prior PR270 hypotheses and the comparison run's unapplied theta-min diagnostics remain unaccepted.
