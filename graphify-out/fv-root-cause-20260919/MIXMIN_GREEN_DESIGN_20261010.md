# GREEN design review: minimize the current gradient-mixture merit

**Scope:** design and saved-receipt review only. This review used the frozen PR270 comparison archive and a small analytic synthetic check. It did not invoke the FV model, generate production gradients/HVPs, run a guard, or alter production code.

## Review decision

The proposed baseline-only mode is mathematically coherent if it treats the current carried theta and the working minimizer theta-star as separate state, requires a resolved one-sided gradient jump and a strict interior theta-star at every evaluated point, and uses freshly recomputed theta-star for each candidate and the final closure. The envelope derivative and the full residual derivative given in the proposal are correct on one smooth selected-branch neighborhood. The mode should remain a local merit search, not be described as minimizing native J or as establishing an optimum.

The single-point comparison does not validate a full-GN arm; that is outside this review. The directional HVPs needed for this baseline merit mode are exactly two at the base point, and the existing 24-row / GN construction is unnecessary.

## Frozen base evidence

The actual PR270 experiment plan is `TANGENT_PRECONDITION_COMPARISON_PLAN_20261009.json`, SHA-256 `be2886f2686cd04e71d3d244967da378bae9c22e39168c5f22bd85e7801169a4`; its 131 source pins and 116 archive pins match the saved plan. Its producer/input continuation plan is a separate object, SHA-256 `d5c4fa93ff4c3fddedb6d458fedea916c0f06d32edfe8bf55d2fd44449ac9d10`, with 129 source and 105 archive pins. These counts and hashes refer to different plans.

The PR270 raw comparison is `tangent_precondition_comparison_20261009_attempt1/step.json`, SHA-256 `c64480cfdc8b011b63b8bec4fd0e704187b28657d47e5e51b63f606406e0c4cf`; its lossless gzip, run, and resource receipts are pinned by `PRECOND_ARCHIVE_20261009.json`. The correct 0da base fields come from the uniquely selected `baseline_tangent` trial and its `final_repeat`, not from the raw top-level scalar fields. I checked the selected trial and final repeat: the 26-control arrays hash to `0da57dbc5a4c101c2f24204835a746514b110f4548fe64de40455f4478dfbed1`; both side-gradient arrays are byte-equal; native J, carried theta, and carried F² agree exactly; and the `-1/+1` trace signatures both equal `8c730e5538bcb93696e46a97d9c978f27779ed0d5267eeada514860b2e681d37`. The repeat records true native-objective, side-objective, merit, face, branch-pair, trace, source, fixed-input, runtime, and deadline closure flags.

| Quantity at the frozen 0da point | Saved value |
|---|---:|
| Carried theta | 0.3298981720696071 |
| Native J | 0.06124393564803218 |
| Carried-mixture F² | 0.005132555144362135 |
| `theta_star` from the saved `g-`, `g+` | 0.3277376621747043 |
| Minimized gradient-mixture F² at `theta_star` | 0.005132174540958916 |
| `||g+ - g-||` | 0.2855486254674688 |
| `g(theta_star) · (g+ - g-)` | -4.34e-19 |

The saved-array recomputation lowers the mixture F² by `3.806034032e-7`, or `0.0074155%` relative to the carried-mixture F². This is a change of mixture weight at the same control, not a new control step, an optimizer iteration, or a native-J improvement. The raw top-level `current_native_objective` and `accepted_F_squared` still contain the pre-comparison 33cb values even though `current_control_sha256` points to 0da; the 0da trial and final-repeat records are the authoritative base receipts for this review.

## Mathematical check

Let `g-(x)` and `g+(x)` be the two selected-side native-J gradients, `j(x)=g+(x)-g-(x)`, and `q(x)` the selected-face residual. For the fixed residual convention with `q_s=0.84`, define

```text
g_theta(x) = g-(x) + theta j(x)
R_theta(x) = [g_theta(x), q(x)/q_s]
Psi(x, theta) = 1/2 ||R_theta(x)||^2.
```

Because `q` is independent of theta, minimizing `Psi` over theta is the same scalar quadratic as minimizing `||g_theta||²`. If `j·j` is finite and resolved, the unconstrained minimizer is `-g-·j/(j·j)`. This mode must refuse unless that value is strictly inside `(0,1)`; clipped endpoints have a different active-bound derivative and need a separately audited mode. The resolved-jump threshold should be scaled against both side-gradient norms and FP64 roundoff, so a tiny nominal denominator cannot produce a numerically arbitrary theta.

At the interior minimizer, `g·j=0`. Along a differentiable fixed-branch path with velocity `d`, let `h-=H-d`, `h+=H+d`, and `h_mix=(1-theta_star)h-+theta_star h+`. Differentiating the first-order condition `g·j=0` gives

```text
theta_star' = -[j·h_mix + g·(h+ - h-)] / (j·j).
```

Thus the derivative of the residual vector is

```text
DF = [h_mix + j theta_star', (n·d)/q_s],
```

where `n=grad(q)`. The scalar envelope derivative is

```text
D Psi = R·DF = g·h_mix + q (n·d)/q_s².
```

The `theta_star'` contribution cancels from this scalar derivative by `g·j=0`, but it must stay in the vector `DF` used to form the linearized merit cap. A local cap
`alpha <= -R·DF / ||DF||²` is valid only when the computed slope is finite and strictly negative beyond a scale-based roundoff budget; combine it with `alpha <= 1` and the actual chart-radius cap. Ratios should be evaluated in normalized form to avoid squaring an extreme norm.

An analytic quadratic two-branch synthetic check compared central finite differences to the formulas above. The finite-difference theta-star derivative matched the analytic value to `2.9e-12`; the residual-vector derivative maximum component error was `2.7e-12`; and the merit slope matched to `1.8e-12`. This verifies the algebra in a smooth synthetic case only, not the FV derivatives or a live control point.

## Candidate and transaction contract

Use the Q=0 chart and `d=-P g(theta_star)` at the base. Require the side-gradient jump to be supported by the selected face normal within a construction-scaled FP64 budget, as the existing tangent code does at [fv_point_3h_tangent_continuation.py](/Users/yhlee/ADVAR/examples/weather_scenarios/fv_point_3h_tangent_continuation.py:118). Compute exactly two fresh current-point side HVPs along this `d`. Do not build row Jacobians, a full Hessian, a GN surrogate, a dense solve, or PCG in this mode.

For every candidate control produced by the bounded chart path, freshly evaluate both side gradients, native J, the common-face/branch-pair and face checks, then recompute its own strict-interior `theta_star` and minimized F². Apply native-J Armijo against the exact 0da base J and minimized-merit Armijo against the base merit recomputed at the base `theta_star`. A linearly transported theta may be logged as a model diagnostic only: do not pre-refuse a candidate because this diagnostic falls outside `[0,1]`, and never substitute it for the candidate's freshly minimized theta. Conversely, if a candidate's actual minimizer is at/outside a bound or its jump is unresolved, refuse that candidate.

The last fresh closure must recompute both side gradients and native J at the exact proposed control; recompute theta-star and minimized F²; verify those values against the proposal at scale-aware tolerances; and repeat trace, face, pair, source, fixed-input, runtime, and deadline checks. Commit at most one accepted endpoint atomically. If base admission, all candidates, or final closure refuses, preserve the original 0da control and its carried theta `0.3298981720696071`; do not save `theta_star` alone or count it as a step. If closure passes, store the candidate control and its verified candidate `theta_star` together.

The proposed resource envelope is internally sensible for this scope: one root-owned guarded attempt, two HVPs total, no Jacobian-row ledger, at most 16 line-search candidates, one commit, 240-second inner / 300-second outer ceilings, and a 1 GiB RSS cap. Candidate evaluations and the final repeat still need explicit counters and deadline checks; refusing before the guard is not a production evaluation.

## Qualification

The saved 0da endpoint is a valid immutable input receipt, but a future run must freeze and pin the exact source and inputs it executes. Do not infer a new run's source provenance from the old carried receipt. This design supports only the all-detected fixed point and the strict-interior, resolved-jump, common-face branch neighborhood. It does not cover clipped theta bounds, branch-transition derivatives, mixed/masked observation contracts, the later full-GN comparison, convergence, global curvature, a minimizer of J, a forecast score, or response.
