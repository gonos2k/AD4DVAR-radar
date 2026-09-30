# R4-R finite-offset tangent-gradient diagnostic, attempt 1

The previous guarded face-event run located `q_y[2,0]=0` on the two
first accepted PR #209 chords, but did not save the full control
gradient at its eight finite off-event samples. This one-shot read-only
diagnostic asks whether a substantial objective-gradient component
remains **tangent** to that face-zero surface. It does not optimize,
differentiate the objective at the switch, or issue an adjoint or
response. Preserve the original 4×5, 58-valid/2-missing observations,
prior, verification field, 54-stage branch signatures and thresholds.

Require SHA256 of the PR #225 child
`7b0798414c2f7036189db0afef5cbf101b53c8903c29acf4079b4de0f2b0ca1d`,
its evidence manifest
`df72467af26217534a59435c9f689b337d2d2f96cf13c48c078b803af9f6bec4`,
the PR #209 archive manifest SHA256
`d80362f8acbb0feb1c82d60d5e9cde61a4863564d7f8f64e49467ba39d12cddd`,
and all listed raw/source/fixed-input hashes before and after. Rebuild
the eight off-event controls from the exact archived 26-component
endpoints and recorded `t`; require their tensor SHA256 and event
coordinates to match PR #225. Only the first two accepted chords and
the two declared offsets `2^-8`, `2^-10` on each side are in scope.

Use existing **optimizer control coordinates** `c` with Euclidean
metric. This is a declared coordinate-dependent numerical diagnostic,
not an invariant physical norm; retain field-20, flow-5 and growth-1 block
norms so unit/scale effects are visible. Compute the smooth face-flux
normal `n = grad_c q_y[2,0](c_event)` from the production bounded
coefficient, streamfunction and face-flux functions. Use a **tensor-valued**
face scalar throughout this AD path; the prior `_qy_20()` helper returns
a Python float and must not be used for differentiation. Hold the
recorded event parameter fixed and differentiate with respect to the
ambient control. Flux is smooth through zero even though minmod/upwind
choices switch there. Require finite `n` and
`||n||_2 > 128*eps64*max(1, ||n||_2)` before normalizing. This is a
fixed-coordinate numerical zero guard, not a scale-invariant
regularity certificate; report the normal magnitude and flux scale. It is
permitted to differentiate this face map at `c_event`; **do not**
differentiate `J`, the FV trajectory or a score at that event. Require
finite `n` with resolved nonzero norm and report its block entries.

For each of the eight fixed finite controls, compute
`g=grad_c J(c,p)`, separately from the event. Record full finite `g`,
`||g||_2`, `||g||_inf`, the unit-normal slope `g·n_hat`, and the
Euclidean tangent projection `g_T=g-(g·n_hat)n_hat` with its norm and
field/flow/growth block norms. Independently recheck the archived
gradient-2 and gradient-infinity scalars to absolute tolerance
`128*eps64*max(1, |reported scalar|)`;
if they disagree, invalidate the diagnostic rather than adjust the
threshold. The previous strict branch/margin status remains attached
to each sample; it is not recomputed or upgraded.

At each chord and offset, compare **only that chord's own** left/right
sampled gradients. The minimum-norm point on their line segment is
`g_L + alpha*(g_R-g_L)` with
`alpha=clip(-g_L·(g_R-g_L)/||g_R-g_L||²,0,1)`; report the selected
alpha and Euclidean distance to zero. If the difference norm is at
most `128*eps64*max(1, ||g_L||_2, ||g_R||_2)`, report a degenerate pair
and do not divide by it.
Also report each side's near/far tangent-vector change. These are
finite-offset sampled-gradient hulls, **not** the Clarke
subdifferential or a proof of limiting-gradient convergence. Do not
combine gradients from the two distinct event controls.

Run one serial child under a 120-second wall trigger and sampled
1-GiB child-RSS trigger, writing only a fresh attempt directory outside
the PR #209/#225 raw archives. Pin the reviewed plan and probe/runner
source SHA256 for the guarded launch; separate process completion,
numeric diagnostic status and response eligibility. No second offset
sequence or post-hoc replacement is authorized in this attempt.

A resolved nonzero tangent component at both offsets can motivate a
subsequent branch-aware search **along** the face-zero manifold; a
small component would instead motivate a separate constrained/nonsmooth
stationarity study. These are off-event gradients projected onto the
event-point tangent **hyperplane**, not gradients of `J` restricted to
the curved face-zero surface. Neither result closes R4-R or proves that the face
caused the old refusal, that a smooth interior root exists, or that
local implicit sensitivity is available. All archived side samples
failed the existing `1e-4` response face-margin gate.
