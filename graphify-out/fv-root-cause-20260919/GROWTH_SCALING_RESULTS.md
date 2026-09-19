# Small-growth numerical resolution

The FV integrating factor is still exp(g). For |g| < 1/8, evaluate its product
with a nonnegative state as `addcmul(q, q, expm1(g))`, mathematically
q + q(exp(g)-1). This retains the small increment before the final addition;
it avoids first rounding exp(g) near one. No universal backend FMA claim is made.
Outside this interval the original exponential product is retained. The small
negative branch subtracts less than 12% of q, avoiding severe cancellation.

Scalar exponential finiteness checks, scalable-boundary masks, large-trace
splitting, inverse-budget products, and the support operator are unchanged.
The new product is used at the same physical growth locations as before.

## Distinguishing example

For CPU FP64, q=1.5, g=0.49*epsilon, zero flux:

- exp(g) rounds to 1; the previous product returns 1.5.
- A 100-digit Decimal evaluation rounds to 1.5000000000000002.
- The revised FV step returns that value. First and second derivatives are
  checked separately. Existing extreme-growth/subnormal tests remain required.

## Large-state evidence

The corrected temporary candidate was evaluated with the same B=y[0] contract
in both gradient evaluation and root refinement, using retained checkpoints.
It used 45.42 s / 1.694 GB sampled RSS.

| State | Candidate initial max gradient | After refinement |
|---|---:|---:|
| Nominal 240x240 | 2.07812e-8 | 5.73043e-10 |
| h=0.001 dBZ | 9.46748e-9 | 9.46748e-9 (already within gate) |

The unchanged response gate is max|gradient| <= 1e-8. The nominal correction
used two exact Hessian products and an actual-residual PCG check. This is local
stationarity evidence, not global positive definiteness or a minimum proof.
The full adjoint and finite-reanalysis comparison under the revised arithmetic
are separate checks, currently in progress.

The first temporary candidate driver incorrectly reused the nominal background
for the perturbed root solve; it was terminated and its calculations are excluded.
The corrected driver constructs one frozen contract per observation set and
passes it to both gradient evaluation and refinement. Product code was not
changed during that invalid diagnostic.

Validation available before full-response rerun: 28 numerical-range tests and
4 replay/JVP/VJP/mixed-second-derivative tests passed. See the current extension
checklist for further affected tests and completion status.

## Applied implementation checks

The applied 240x240 nominal rerun completed in 490.42 s / 1.947 GB sampled RSS:
max gradient 5.73043e-10, actual adjoint relative residual 4.28392e-11,
41 Hessian products. See rotation240_stable_response_18.json; it records current
source hashes and saved tensor/checkpoint hashes. Finite perturbations are still
separate from this nominal result.

Affected suite: 117 passed, one finite-difference regression initially failed.
The endpoint solve residual was amplified by differencing: errors grew from
3.71e-8 (h=2e-4) to 3.48e-7 (h=2.5e-5). Using existing exact root refinement
at 1e-10 for the test endpoints reduced the h=1e-4 slope error to 4.01e-10;
the corrected test passed in 43.84 s without relaxing comparison tolerance.
The added negative-growth boundary case and existing positive-growth case both
passed (2 tests, 1.64 s), including budget, JVP/VJP and finite difference checks.

Prescribed-flow grid/JVP convergence rerun under current arithmetic completed
in 6.06 s / 331 MB, recorded in fv_grid_convergence_stable_growth.json.
These results do not establish general P1 or real-data predictive skill.

## Positive finite reanalyses

Both retained positive perturbations now pass the unchanged max-gradient gate:

| h (dBZ) | Actual score change | Linear change | Remainder | max gradient |
|---:|---:|---:|---:|---:|
| 0.001 | 5.3930307184e-6 | 2.2684523069e-6 | 3.1245784114e-6 | 9.46748e-9 |
| 0.0005 | 1.9154192749e-6 | 1.1342261535e-6 | 7.8119312138e-7 | 1.42878e-10 |

Remainder ratio is 3.999751567 on halving h, consistent with an O(h^2)
Taylor remainder. The first-order remainder is nevertheless large relative to
the linear term at these h; do not call these finite effects accurately linear.
One-sided Richardson slope is 0.002268646381, versus adjoint 0.002268452307.
A matched negative reanalysis is the next independent central-slope check.
The positive run used 353.50 s / 1.927 GB sampled peak RSS.

## Matched central reanalysis

At h=0.0005 dBZ, both endpoints were independently stationarity-checked:
positive max gradient 1.42878e-10, negative 2.93110e-10. Central slope is
0.002268352553, versus exact-adjoint slope 0.002268452307: absolute difference
9.97536e-8, relative difference 4.39743e-5 (0.00440%). Negative score change
is -3.52933e-7; the positive/negative quadratic remainders are respectively
7.81193e-7 and 7.81293e-7. Both branch margins exceed 0.274.

Together with the positive-step factor-3.99975 remainder reduction, this supports
the local derivative in this declared smooth direction on the synthetic rotation.
It is not an all-directions conditioning proof, global minimum proof, accurate
linear prediction for arbitrary finite perturbations, or general-FV claim.
No tolerance was relaxed. The central run used 342.96 s / 1.892 GB sampled RSS.
See rotation240_stable_central_0.0005.json and the original HTML response panel.
