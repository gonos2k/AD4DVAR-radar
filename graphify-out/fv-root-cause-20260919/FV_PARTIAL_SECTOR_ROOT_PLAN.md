# R4-R one current-source sector-aware partial FV root feasibility run

The archived two-hole collocated partial-input attempt 3 refused all 16
final Newton candidates under its fixed GN signature and `1e-4`
face-margin gate. Its 3 signature / 13 face-margin counts are
**first-failing checks**; the 13 candidates were not signature-tested
and no finite objective/gradient Armijo trial ran. That refusal and
the R4-D support decision remain unchanged.

This is **one new current-source, root-only experiment**, not a replay
of the historical measured source. The historical preflight problem
digest was `ffc5a82cb9e4cda7667378b1932c6703e61a6a0c2d16d37ca1c6268b2f860e48`;
the current versioned problem digest is
`838fcf77a43b209b53dfe62f6a1657d0f8eba5a41797e431451059f3f8d18158`.
Current and archived observation/mask/whitener/parameter/warm-control/
verification/direction tensor SHA256 fields, observation counts,
missing indices, warm objective/maximum gradient and strict warm
branch signature/margins must agree. The changed problem digest and
source hashes must be reported explicitly; do not call this exact
historical binary/source reproduction. Pin the historical preflight
JSON SHA256 `4a2fd9c57e6f271e0505e6ffb9bb7bc9bca66fff724ce0d6e885df90d7a7657b`
and attempt-3 manifest SHA256
`6d859284dbb737910e2a94252e37cf87322924cae2fa9e88381f9e29dd18a5a3`.

Use the existing product `solve_analysis` once from the same frozen
warm control, without changing the partial objective, prior, fixed
verification field, two missing indices, covariance/whitener or
direction. Record the returned GN reason, control and objective.
Before exact Newton, audit the GN point with all 26 exact gradient-JVP
Hessian columns: finite values, relative antisymmetry `<=1e-10`,
positive minimum eigenvalue and minimum/maximum eigenvalue ratio
`>sqrt(eps64)`. If the current GN point is not locally SPD, classify
`seed_curvature_refused` with no Newton or response.

For at most 8 Newton steps, use the unchanged exact-HVP/PCG refiner
with PCG actual relative residual `<=1e-10`, max 104 iterations per
linear solve, and max 16 backtracks. Every trial branch callback must
complete the roundoff-aware 54-stage minmod oracle and measure finite,
**positive** scaled slope and face margins. Do not enforce the GN
signature or terminal `1e-4` margins on an intermediate root-search
candidate. Retain the full limiter-choice plus face-sign signature.
For the same full signature, return `None` from the optional acceptance
hook and use the existing normalized gradient-merit Armijo rule. For
**every changed-signature finite candidate**, return an explicit
decision: accept only if measured `Phi=0.5||grad_c J||²` falls by more
than `128*eps64*max(|Phi_old|,|Phi_new|,tiny64)`. Never reuse the old
branch's Newton slope as a cross-signature acceptance certificate.
After an accepted switch, the next Newton iteration relinearizes on
the new branch. Endpoint checks do not certify smoothness along the
segment, nor global convergence or a physical minimum.

Atomically record candidate control hashes, actual J, gradient norm
and maximum, full-signature digest, measured margins, switch rule,
decision/rejection reason, and PCG/HVP/true linear-residual evidence.
Unknown callback/invariant errors are execution failures, never
scientific refusals. After convergence, independently recompute
`||grad_c J||_inf<1e-10`, the full exact Hessian audit and a fresh
54-stage branch. A pointwise SPD root with either scaled margin
`<=1e-4` is `root_low_margin` and remains response-ineligible;
only both margins `>1e-4` can be `root_margin_qualified`. Both still
have `response_validation=not_performed` here. Do **not** calculate
an adjoint, VJP, tangent or signed nonlinear endpoints in this run.

Launch exactly one serial child under a 600-second wall and sampled
1-GiB child-RSS cap. Parent rechecks plan/source/input and historical
preflight/manifest identities before/after, command/PID/exit,
sampled resource status, and root/refusal-specific evidence. A
scientific refusal remains a refusal; no post-hoc seed, threshold,
budget or objective change is allowed in this attempt. A successful
root is at most a candidate for a **later** separately guarded
partial-observation response experiment.
