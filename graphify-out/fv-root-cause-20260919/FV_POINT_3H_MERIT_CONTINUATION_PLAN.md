# Bounded relinearized joint-merit continuation from the PR #221 endpoint

Start only from the accepted PR #221 endpoint in
`point_3h_merit_step_attempt1/point_3h_merit_step.json` (raw SHA256
`024b89d346ae0136542888527d3bb528d7f638c1e3be44f3aa7cdb2232d10f3a`):
26-control SHA256
`9f8d3ae0d77d8b24f1565ab622cf8fe7bbfd83766fbd67964a6a5bd52920a078`,
strict branch signature
`c1d60b87479fea5937ae44d4740cb0de3adf10e62fc50018cab69abb12ca47bb`,
J `0.9319504904669574`, Phi `2.6187151084118883`, gradient maximum
`1.2605349625093007`. The segment that reached it was not
certified. Keep the same PR #204 4×5 point problem, 13 fixed
parameters, external background, terminal score definition/target,
0/10/20-minute observations, 18 future leads and boundaries. Pin
the PR #221 evidence SHA256
`df026542d85910a61a9181ae46df8f5c7e6ac8d8bfa80dbc318e33d429b85a8a`
and current source/plan identities.

This is a **bounded exploratory continuation**, not a root or
response API. In one guarded child reconstruct the original problem,
replace only the starting control with the saved candidate after
hash/shape/dtype checks, then run at most **8 accepted epochs** and
**64 total endpoint trials** (at most 8 in each epoch). At each
current point independently require a complete strict 3,600-stage
branch. Use a separate no-grad forecast pass to collect complete
minimum normalized slope/active-limiter-gap/face margins before
claiming that branch's diagnostics; if incomplete, refuse rather
than assigning minima. Compute finite analysis objective J, exact
control gradient g, Phi=`||g||²/2`, and gradient maximum.

Before **each** direction compute fresh `Hg=JVP_c(grad_c J;g)` and
fresh `H^Tg=VJP_c(grad_c J;g)` at that same current control and fixed
p. Require both finite and
`||Hg-H^Tg||_2/max(||Hg||_2,||H^Tg||_2,tiny) <=1e-8` for CPU FP64.
This is a one-vector transpose consistency check, **not** a full SPD
or symmetry proof. Use only `d=-Hg/||Hg||_2` in the current latent
coordinates, after finite nonzero norm and
`g^THg >128*eps*||g||_2||Hg||_2`. Record the true stationarity-merit
slope `(H^Tg)^Td`, as well as J slope `g^Td`, and require both finite
negative. If any gate fails, stop with a distinct numerical refusal;
no fallback direction, Hessian shift or hidden policy change.

For each epoch try only `alpha=0.02*2^-k`, `k=0,...,7` in order.
Every finite trial must first pass the **original full strict branch
oracle** at its endpoint, then receive fresh finite J, g, Phi and
gradient maximum. Require measured J and Phi reductions above their
separate `128*eps*max(abs(seed_value),tiny)` floors. If the candidate
signature equals the **current epoch** signature, also require both
Armijo tests (`c1=1e-4`) using that epoch's fresh slopes; if it
differs, use actual two-scalar decrease only. Recompute the
`branch_changed` flag from the two signatures, not a child-reported
boolean. Record all refusals and one accepted endpoint at most per
epoch. After any acceptance, discard that epoch's HVP/VJP and
relinearize on the new strict branch before another direction. No
endpoint equality or switch certifies the connecting segment.

Stop if gradient maximum at an accepted point is below `1e-4` only
after a fresh complete branch-and-margin pass at that endpoint, then
label it **handoff candidate only**; do not call it stationary or
response-eligible. Preserve each epoch's own branch/margin summary,
and bind its starting control, J, Phi and signature to the prior
accepted endpoint; epoch zero must match the pinned PR #221 candidate.
Otherwise stop on 8 accepted epochs, 64 total
trials, a direction/branch/merit refusal, or an internal 540-second
phase budget. The resource guard is one **600-second wall-limit
trigger** plus reap grace and **sampled 1 GiB child RSS**. It may
terminate a single long operation after the internal budget check;
preserve raw partial progress and do not retry or replace the seed.
Checkpoint after each trial and accepted epoch. Report actual child,
guard and parent times and sampled memory; phase times need not sum
to total unless every operation is instrumented. The parent performs
only source/archive/input/plan/guard/result checks, no FV work.

Even a very small final gradient is only a search candidate. A
separately budgeted terminal handoff must certify the final point's
own strict branch and margins, gradient `<1e-10`, exact **unshifted**
Hessian suitability, original-H adjoint true residual, full
13-parameter VJP and two signed nonlinear endpoint pairs before
issuing a sensitivity. No physical-skill or finite-impact claim
follows from this exploration. Run focused/fake-guard and affected
tests, typecheck, incremental Graphify refresh and independent
GREEN/RED prelaunch review before launching the single FV child.
